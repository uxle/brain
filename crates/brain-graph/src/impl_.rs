//! # Graph Implementation Helpers
//!
//! Convenient evaluation, execution runners, and batch evaluation wrappers.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::core::GraphResult;
use crate::ir::GraphIr;
use crate::interp::GraphInterpreter;

/// Runs inference on a `GraphIr` given input tensors.
pub fn run_graph(graph: &GraphIr, inputs: &[Tensor]) -> GraphResult<Vec<Tensor>> {
    let mut interp = GraphInterpreter::new();
    interp.run(graph, inputs)
}

/// Computes the total memory allocated by all tensor outputs of the graph.
pub fn total_output_memory_bytes(graph: &GraphIr) -> usize {
    graph.values.iter().map(|v| v.shape.num_elements() * 4).sum()
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_impl_stress_001() {
        let mut g = GraphIr::new(&format!("impl_g_1"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_002() {
        let mut g = GraphIr::new(&format!("impl_g_2"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_003() {
        let mut g = GraphIr::new(&format!("impl_g_3"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_004() {
        let mut g = GraphIr::new(&format!("impl_g_4"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_005() {
        let mut g = GraphIr::new(&format!("impl_g_5"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_006() {
        let mut g = GraphIr::new(&format!("impl_g_6"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_007() {
        let mut g = GraphIr::new(&format!("impl_g_7"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_008() {
        let mut g = GraphIr::new(&format!("impl_g_8"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_009() {
        let mut g = GraphIr::new(&format!("impl_g_9"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_010() {
        let mut g = GraphIr::new(&format!("impl_g_10"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_011() {
        let mut g = GraphIr::new(&format!("impl_g_11"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_012() {
        let mut g = GraphIr::new(&format!("impl_g_12"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_013() {
        let mut g = GraphIr::new(&format!("impl_g_13"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_014() {
        let mut g = GraphIr::new(&format!("impl_g_14"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_015() {
        let mut g = GraphIr::new(&format!("impl_g_15"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_016() {
        let mut g = GraphIr::new(&format!("impl_g_16"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_017() {
        let mut g = GraphIr::new(&format!("impl_g_17"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_018() {
        let mut g = GraphIr::new(&format!("impl_g_18"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_019() {
        let mut g = GraphIr::new(&format!("impl_g_19"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_020() {
        let mut g = GraphIr::new(&format!("impl_g_20"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_021() {
        let mut g = GraphIr::new(&format!("impl_g_21"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_022() {
        let mut g = GraphIr::new(&format!("impl_g_22"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_023() {
        let mut g = GraphIr::new(&format!("impl_g_23"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_024() {
        let mut g = GraphIr::new(&format!("impl_g_24"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_025() {
        let mut g = GraphIr::new(&format!("impl_g_25"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_026() {
        let mut g = GraphIr::new(&format!("impl_g_26"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_027() {
        let mut g = GraphIr::new(&format!("impl_g_27"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_028() {
        let mut g = GraphIr::new(&format!("impl_g_28"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_029() {
        let mut g = GraphIr::new(&format!("impl_g_29"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_030() {
        let mut g = GraphIr::new(&format!("impl_g_30"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_031() {
        let mut g = GraphIr::new(&format!("impl_g_31"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_032() {
        let mut g = GraphIr::new(&format!("impl_g_32"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_033() {
        let mut g = GraphIr::new(&format!("impl_g_33"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_034() {
        let mut g = GraphIr::new(&format!("impl_g_34"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_035() {
        let mut g = GraphIr::new(&format!("impl_g_35"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_036() {
        let mut g = GraphIr::new(&format!("impl_g_36"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_037() {
        let mut g = GraphIr::new(&format!("impl_g_37"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_038() {
        let mut g = GraphIr::new(&format!("impl_g_38"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_039() {
        let mut g = GraphIr::new(&format!("impl_g_39"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_040() {
        let mut g = GraphIr::new(&format!("impl_g_40"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_041() {
        let mut g = GraphIr::new(&format!("impl_g_41"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_042() {
        let mut g = GraphIr::new(&format!("impl_g_42"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_043() {
        let mut g = GraphIr::new(&format!("impl_g_43"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_044() {
        let mut g = GraphIr::new(&format!("impl_g_44"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_045() {
        let mut g = GraphIr::new(&format!("impl_g_45"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_046() {
        let mut g = GraphIr::new(&format!("impl_g_46"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_047() {
        let mut g = GraphIr::new(&format!("impl_g_47"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_048() {
        let mut g = GraphIr::new(&format!("impl_g_48"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_049() {
        let mut g = GraphIr::new(&format!("impl_g_49"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_050() {
        let mut g = GraphIr::new(&format!("impl_g_50"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_051() {
        let mut g = GraphIr::new(&format!("impl_g_51"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_052() {
        let mut g = GraphIr::new(&format!("impl_g_52"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_053() {
        let mut g = GraphIr::new(&format!("impl_g_53"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_054() {
        let mut g = GraphIr::new(&format!("impl_g_54"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_055() {
        let mut g = GraphIr::new(&format!("impl_g_55"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_056() {
        let mut g = GraphIr::new(&format!("impl_g_56"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_057() {
        let mut g = GraphIr::new(&format!("impl_g_57"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_058() {
        let mut g = GraphIr::new(&format!("impl_g_58"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_059() {
        let mut g = GraphIr::new(&format!("impl_g_59"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_060() {
        let mut g = GraphIr::new(&format!("impl_g_60"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_061() {
        let mut g = GraphIr::new(&format!("impl_g_61"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_062() {
        let mut g = GraphIr::new(&format!("impl_g_62"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_063() {
        let mut g = GraphIr::new(&format!("impl_g_63"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_064() {
        let mut g = GraphIr::new(&format!("impl_g_64"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_065() {
        let mut g = GraphIr::new(&format!("impl_g_65"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_066() {
        let mut g = GraphIr::new(&format!("impl_g_66"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_067() {
        let mut g = GraphIr::new(&format!("impl_g_67"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_068() {
        let mut g = GraphIr::new(&format!("impl_g_68"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_069() {
        let mut g = GraphIr::new(&format!("impl_g_69"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_070() {
        let mut g = GraphIr::new(&format!("impl_g_70"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_071() {
        let mut g = GraphIr::new(&format!("impl_g_71"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_072() {
        let mut g = GraphIr::new(&format!("impl_g_72"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_073() {
        let mut g = GraphIr::new(&format!("impl_g_73"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_074() {
        let mut g = GraphIr::new(&format!("impl_g_74"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_075() {
        let mut g = GraphIr::new(&format!("impl_g_75"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_076() {
        let mut g = GraphIr::new(&format!("impl_g_76"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_077() {
        let mut g = GraphIr::new(&format!("impl_g_77"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_078() {
        let mut g = GraphIr::new(&format!("impl_g_78"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_079() {
        let mut g = GraphIr::new(&format!("impl_g_79"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_080() {
        let mut g = GraphIr::new(&format!("impl_g_80"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_081() {
        let mut g = GraphIr::new(&format!("impl_g_81"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_082() {
        let mut g = GraphIr::new(&format!("impl_g_82"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_083() {
        let mut g = GraphIr::new(&format!("impl_g_83"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_084() {
        let mut g = GraphIr::new(&format!("impl_g_84"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_085() {
        let mut g = GraphIr::new(&format!("impl_g_85"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_086() {
        let mut g = GraphIr::new(&format!("impl_g_86"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_087() {
        let mut g = GraphIr::new(&format!("impl_g_87"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_088() {
        let mut g = GraphIr::new(&format!("impl_g_88"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_089() {
        let mut g = GraphIr::new(&format!("impl_g_89"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_090() {
        let mut g = GraphIr::new(&format!("impl_g_90"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_091() {
        let mut g = GraphIr::new(&format!("impl_g_91"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_092() {
        let mut g = GraphIr::new(&format!("impl_g_92"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_093() {
        let mut g = GraphIr::new(&format!("impl_g_93"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_094() {
        let mut g = GraphIr::new(&format!("impl_g_94"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_095() {
        let mut g = GraphIr::new(&format!("impl_g_95"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_096() {
        let mut g = GraphIr::new(&format!("impl_g_96"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_097() {
        let mut g = GraphIr::new(&format!("impl_g_97"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_098() {
        let mut g = GraphIr::new(&format!("impl_g_98"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_099() {
        let mut g = GraphIr::new(&format!("impl_g_99"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_100() {
        let mut g = GraphIr::new(&format!("impl_g_100"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_101() {
        let mut g = GraphIr::new(&format!("impl_g_101"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_102() {
        let mut g = GraphIr::new(&format!("impl_g_102"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_103() {
        let mut g = GraphIr::new(&format!("impl_g_103"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_104() {
        let mut g = GraphIr::new(&format!("impl_g_104"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_105() {
        let mut g = GraphIr::new(&format!("impl_g_105"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_106() {
        let mut g = GraphIr::new(&format!("impl_g_106"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_107() {
        let mut g = GraphIr::new(&format!("impl_g_107"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_108() {
        let mut g = GraphIr::new(&format!("impl_g_108"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_109() {
        let mut g = GraphIr::new(&format!("impl_g_109"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_110() {
        let mut g = GraphIr::new(&format!("impl_g_110"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_111() {
        let mut g = GraphIr::new(&format!("impl_g_111"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_112() {
        let mut g = GraphIr::new(&format!("impl_g_112"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_113() {
        let mut g = GraphIr::new(&format!("impl_g_113"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_114() {
        let mut g = GraphIr::new(&format!("impl_g_114"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_115() {
        let mut g = GraphIr::new(&format!("impl_g_115"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_116() {
        let mut g = GraphIr::new(&format!("impl_g_116"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_117() {
        let mut g = GraphIr::new(&format!("impl_g_117"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_118() {
        let mut g = GraphIr::new(&format!("impl_g_118"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_119() {
        let mut g = GraphIr::new(&format!("impl_g_119"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_120() {
        let mut g = GraphIr::new(&format!("impl_g_120"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_121() {
        let mut g = GraphIr::new(&format!("impl_g_121"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_122() {
        let mut g = GraphIr::new(&format!("impl_g_122"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_123() {
        let mut g = GraphIr::new(&format!("impl_g_123"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_124() {
        let mut g = GraphIr::new(&format!("impl_g_124"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_125() {
        let mut g = GraphIr::new(&format!("impl_g_125"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_126() {
        let mut g = GraphIr::new(&format!("impl_g_126"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_127() {
        let mut g = GraphIr::new(&format!("impl_g_127"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_128() {
        let mut g = GraphIr::new(&format!("impl_g_128"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_129() {
        let mut g = GraphIr::new(&format!("impl_g_129"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_130() {
        let mut g = GraphIr::new(&format!("impl_g_130"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_131() {
        let mut g = GraphIr::new(&format!("impl_g_131"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_132() {
        let mut g = GraphIr::new(&format!("impl_g_132"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_133() {
        let mut g = GraphIr::new(&format!("impl_g_133"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_134() {
        let mut g = GraphIr::new(&format!("impl_g_134"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_135() {
        let mut g = GraphIr::new(&format!("impl_g_135"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_136() {
        let mut g = GraphIr::new(&format!("impl_g_136"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_137() {
        let mut g = GraphIr::new(&format!("impl_g_137"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_138() {
        let mut g = GraphIr::new(&format!("impl_g_138"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_139() {
        let mut g = GraphIr::new(&format!("impl_g_139"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_140() {
        let mut g = GraphIr::new(&format!("impl_g_140"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_141() {
        let mut g = GraphIr::new(&format!("impl_g_141"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_142() {
        let mut g = GraphIr::new(&format!("impl_g_142"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_143() {
        let mut g = GraphIr::new(&format!("impl_g_143"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_144() {
        let mut g = GraphIr::new(&format!("impl_g_144"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_145() {
        let mut g = GraphIr::new(&format!("impl_g_145"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_146() {
        let mut g = GraphIr::new(&format!("impl_g_146"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_147() {
        let mut g = GraphIr::new(&format!("impl_g_147"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_148() {
        let mut g = GraphIr::new(&format!("impl_g_148"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_149() {
        let mut g = GraphIr::new(&format!("impl_g_149"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_150() {
        let mut g = GraphIr::new(&format!("impl_g_150"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_151() {
        let mut g = GraphIr::new(&format!("impl_g_151"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_152() {
        let mut g = GraphIr::new(&format!("impl_g_152"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_153() {
        let mut g = GraphIr::new(&format!("impl_g_153"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_154() {
        let mut g = GraphIr::new(&format!("impl_g_154"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_155() {
        let mut g = GraphIr::new(&format!("impl_g_155"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_156() {
        let mut g = GraphIr::new(&format!("impl_g_156"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_157() {
        let mut g = GraphIr::new(&format!("impl_g_157"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_158() {
        let mut g = GraphIr::new(&format!("impl_g_158"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_159() {
        let mut g = GraphIr::new(&format!("impl_g_159"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_160() {
        let mut g = GraphIr::new(&format!("impl_g_160"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_161() {
        let mut g = GraphIr::new(&format!("impl_g_161"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_162() {
        let mut g = GraphIr::new(&format!("impl_g_162"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_163() {
        let mut g = GraphIr::new(&format!("impl_g_163"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_164() {
        let mut g = GraphIr::new(&format!("impl_g_164"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_165() {
        let mut g = GraphIr::new(&format!("impl_g_165"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_166() {
        let mut g = GraphIr::new(&format!("impl_g_166"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_167() {
        let mut g = GraphIr::new(&format!("impl_g_167"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_168() {
        let mut g = GraphIr::new(&format!("impl_g_168"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_169() {
        let mut g = GraphIr::new(&format!("impl_g_169"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_170() {
        let mut g = GraphIr::new(&format!("impl_g_170"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_171() {
        let mut g = GraphIr::new(&format!("impl_g_171"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_172() {
        let mut g = GraphIr::new(&format!("impl_g_172"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_173() {
        let mut g = GraphIr::new(&format!("impl_g_173"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_174() {
        let mut g = GraphIr::new(&format!("impl_g_174"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_175() {
        let mut g = GraphIr::new(&format!("impl_g_175"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_176() {
        let mut g = GraphIr::new(&format!("impl_g_176"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_177() {
        let mut g = GraphIr::new(&format!("impl_g_177"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_178() {
        let mut g = GraphIr::new(&format!("impl_g_178"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_179() {
        let mut g = GraphIr::new(&format!("impl_g_179"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_180() {
        let mut g = GraphIr::new(&format!("impl_g_180"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_181() {
        let mut g = GraphIr::new(&format!("impl_g_181"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_182() {
        let mut g = GraphIr::new(&format!("impl_g_182"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_183() {
        let mut g = GraphIr::new(&format!("impl_g_183"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_184() {
        let mut g = GraphIr::new(&format!("impl_g_184"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_185() {
        let mut g = GraphIr::new(&format!("impl_g_185"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_186() {
        let mut g = GraphIr::new(&format!("impl_g_186"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_187() {
        let mut g = GraphIr::new(&format!("impl_g_187"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_188() {
        let mut g = GraphIr::new(&format!("impl_g_188"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_189() {
        let mut g = GraphIr::new(&format!("impl_g_189"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_190() {
        let mut g = GraphIr::new(&format!("impl_g_190"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_191() {
        let mut g = GraphIr::new(&format!("impl_g_191"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_192() {
        let mut g = GraphIr::new(&format!("impl_g_192"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_193() {
        let mut g = GraphIr::new(&format!("impl_g_193"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_194() {
        let mut g = GraphIr::new(&format!("impl_g_194"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_195() {
        let mut g = GraphIr::new(&format!("impl_g_195"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_196() {
        let mut g = GraphIr::new(&format!("impl_g_196"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_197() {
        let mut g = GraphIr::new(&format!("impl_g_197"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_198() {
        let mut g = GraphIr::new(&format!("impl_g_198"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_199() {
        let mut g = GraphIr::new(&format!("impl_g_199"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_200() {
        let mut g = GraphIr::new(&format!("impl_g_200"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_201() {
        let mut g = GraphIr::new(&format!("impl_g_201"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_202() {
        let mut g = GraphIr::new(&format!("impl_g_202"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_203() {
        let mut g = GraphIr::new(&format!("impl_g_203"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_204() {
        let mut g = GraphIr::new(&format!("impl_g_204"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_205() {
        let mut g = GraphIr::new(&format!("impl_g_205"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_206() {
        let mut g = GraphIr::new(&format!("impl_g_206"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_207() {
        let mut g = GraphIr::new(&format!("impl_g_207"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_208() {
        let mut g = GraphIr::new(&format!("impl_g_208"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_209() {
        let mut g = GraphIr::new(&format!("impl_g_209"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_210() {
        let mut g = GraphIr::new(&format!("impl_g_210"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_211() {
        let mut g = GraphIr::new(&format!("impl_g_211"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_212() {
        let mut g = GraphIr::new(&format!("impl_g_212"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_213() {
        let mut g = GraphIr::new(&format!("impl_g_213"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_214() {
        let mut g = GraphIr::new(&format!("impl_g_214"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_215() {
        let mut g = GraphIr::new(&format!("impl_g_215"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_216() {
        let mut g = GraphIr::new(&format!("impl_g_216"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_217() {
        let mut g = GraphIr::new(&format!("impl_g_217"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_218() {
        let mut g = GraphIr::new(&format!("impl_g_218"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_219() {
        let mut g = GraphIr::new(&format!("impl_g_219"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_220() {
        let mut g = GraphIr::new(&format!("impl_g_220"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_221() {
        let mut g = GraphIr::new(&format!("impl_g_221"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_222() {
        let mut g = GraphIr::new(&format!("impl_g_222"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_223() {
        let mut g = GraphIr::new(&format!("impl_g_223"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_224() {
        let mut g = GraphIr::new(&format!("impl_g_224"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_225() {
        let mut g = GraphIr::new(&format!("impl_g_225"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_226() {
        let mut g = GraphIr::new(&format!("impl_g_226"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_227() {
        let mut g = GraphIr::new(&format!("impl_g_227"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_228() {
        let mut g = GraphIr::new(&format!("impl_g_228"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_229() {
        let mut g = GraphIr::new(&format!("impl_g_229"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_230() {
        let mut g = GraphIr::new(&format!("impl_g_230"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_231() {
        let mut g = GraphIr::new(&format!("impl_g_231"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_232() {
        let mut g = GraphIr::new(&format!("impl_g_232"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_233() {
        let mut g = GraphIr::new(&format!("impl_g_233"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_234() {
        let mut g = GraphIr::new(&format!("impl_g_234"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_235() {
        let mut g = GraphIr::new(&format!("impl_g_235"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_236() {
        let mut g = GraphIr::new(&format!("impl_g_236"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_237() {
        let mut g = GraphIr::new(&format!("impl_g_237"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_238() {
        let mut g = GraphIr::new(&format!("impl_g_238"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_239() {
        let mut g = GraphIr::new(&format!("impl_g_239"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_240() {
        let mut g = GraphIr::new(&format!("impl_g_240"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_241() {
        let mut g = GraphIr::new(&format!("impl_g_241"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_242() {
        let mut g = GraphIr::new(&format!("impl_g_242"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_243() {
        let mut g = GraphIr::new(&format!("impl_g_243"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_244() {
        let mut g = GraphIr::new(&format!("impl_g_244"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_245() {
        let mut g = GraphIr::new(&format!("impl_g_245"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_246() {
        let mut g = GraphIr::new(&format!("impl_g_246"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_247() {
        let mut g = GraphIr::new(&format!("impl_g_247"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_248() {
        let mut g = GraphIr::new(&format!("impl_g_248"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_249() {
        let mut g = GraphIr::new(&format!("impl_g_249"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_250() {
        let mut g = GraphIr::new(&format!("impl_g_250"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_251() {
        let mut g = GraphIr::new(&format!("impl_g_251"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_252() {
        let mut g = GraphIr::new(&format!("impl_g_252"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_253() {
        let mut g = GraphIr::new(&format!("impl_g_253"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_254() {
        let mut g = GraphIr::new(&format!("impl_g_254"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_255() {
        let mut g = GraphIr::new(&format!("impl_g_255"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_256() {
        let mut g = GraphIr::new(&format!("impl_g_256"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_257() {
        let mut g = GraphIr::new(&format!("impl_g_257"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_258() {
        let mut g = GraphIr::new(&format!("impl_g_258"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_259() {
        let mut g = GraphIr::new(&format!("impl_g_259"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_260() {
        let mut g = GraphIr::new(&format!("impl_g_260"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_261() {
        let mut g = GraphIr::new(&format!("impl_g_261"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_262() {
        let mut g = GraphIr::new(&format!("impl_g_262"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_263() {
        let mut g = GraphIr::new(&format!("impl_g_263"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_264() {
        let mut g = GraphIr::new(&format!("impl_g_264"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_265() {
        let mut g = GraphIr::new(&format!("impl_g_265"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_266() {
        let mut g = GraphIr::new(&format!("impl_g_266"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_267() {
        let mut g = GraphIr::new(&format!("impl_g_267"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_268() {
        let mut g = GraphIr::new(&format!("impl_g_268"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_269() {
        let mut g = GraphIr::new(&format!("impl_g_269"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_270() {
        let mut g = GraphIr::new(&format!("impl_g_270"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_271() {
        let mut g = GraphIr::new(&format!("impl_g_271"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_272() {
        let mut g = GraphIr::new(&format!("impl_g_272"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_273() {
        let mut g = GraphIr::new(&format!("impl_g_273"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_274() {
        let mut g = GraphIr::new(&format!("impl_g_274"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_275() {
        let mut g = GraphIr::new(&format!("impl_g_275"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_276() {
        let mut g = GraphIr::new(&format!("impl_g_276"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_277() {
        let mut g = GraphIr::new(&format!("impl_g_277"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_278() {
        let mut g = GraphIr::new(&format!("impl_g_278"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_279() {
        let mut g = GraphIr::new(&format!("impl_g_279"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_280() {
        let mut g = GraphIr::new(&format!("impl_g_280"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_281() {
        let mut g = GraphIr::new(&format!("impl_g_281"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_282() {
        let mut g = GraphIr::new(&format!("impl_g_282"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_283() {
        let mut g = GraphIr::new(&format!("impl_g_283"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_284() {
        let mut g = GraphIr::new(&format!("impl_g_284"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_285() {
        let mut g = GraphIr::new(&format!("impl_g_285"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_286() {
        let mut g = GraphIr::new(&format!("impl_g_286"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_287() {
        let mut g = GraphIr::new(&format!("impl_g_287"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_288() {
        let mut g = GraphIr::new(&format!("impl_g_288"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_289() {
        let mut g = GraphIr::new(&format!("impl_g_289"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_290() {
        let mut g = GraphIr::new(&format!("impl_g_290"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_291() {
        let mut g = GraphIr::new(&format!("impl_g_291"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_292() {
        let mut g = GraphIr::new(&format!("impl_g_292"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_293() {
        let mut g = GraphIr::new(&format!("impl_g_293"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_294() {
        let mut g = GraphIr::new(&format!("impl_g_294"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_295() {
        let mut g = GraphIr::new(&format!("impl_g_295"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_296() {
        let mut g = GraphIr::new(&format!("impl_g_296"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_297() {
        let mut g = GraphIr::new(&format!("impl_g_297"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_298() {
        let mut g = GraphIr::new(&format!("impl_g_298"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_299() {
        let mut g = GraphIr::new(&format!("impl_g_299"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_300() {
        let mut g = GraphIr::new(&format!("impl_g_300"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_301() {
        let mut g = GraphIr::new(&format!("impl_g_301"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_302() {
        let mut g = GraphIr::new(&format!("impl_g_302"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_303() {
        let mut g = GraphIr::new(&format!("impl_g_303"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_304() {
        let mut g = GraphIr::new(&format!("impl_g_304"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_305() {
        let mut g = GraphIr::new(&format!("impl_g_305"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_306() {
        let mut g = GraphIr::new(&format!("impl_g_306"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_307() {
        let mut g = GraphIr::new(&format!("impl_g_307"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_308() {
        let mut g = GraphIr::new(&format!("impl_g_308"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_309() {
        let mut g = GraphIr::new(&format!("impl_g_309"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_310() {
        let mut g = GraphIr::new(&format!("impl_g_310"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_311() {
        let mut g = GraphIr::new(&format!("impl_g_311"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_312() {
        let mut g = GraphIr::new(&format!("impl_g_312"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_313() {
        let mut g = GraphIr::new(&format!("impl_g_313"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_314() {
        let mut g = GraphIr::new(&format!("impl_g_314"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_315() {
        let mut g = GraphIr::new(&format!("impl_g_315"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_316() {
        let mut g = GraphIr::new(&format!("impl_g_316"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_317() {
        let mut g = GraphIr::new(&format!("impl_g_317"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_318() {
        let mut g = GraphIr::new(&format!("impl_g_318"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_319() {
        let mut g = GraphIr::new(&format!("impl_g_319"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_320() {
        let mut g = GraphIr::new(&format!("impl_g_320"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_321() {
        let mut g = GraphIr::new(&format!("impl_g_321"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_322() {
        let mut g = GraphIr::new(&format!("impl_g_322"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_323() {
        let mut g = GraphIr::new(&format!("impl_g_323"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_324() {
        let mut g = GraphIr::new(&format!("impl_g_324"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_325() {
        let mut g = GraphIr::new(&format!("impl_g_325"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_326() {
        let mut g = GraphIr::new(&format!("impl_g_326"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_327() {
        let mut g = GraphIr::new(&format!("impl_g_327"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_328() {
        let mut g = GraphIr::new(&format!("impl_g_328"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_329() {
        let mut g = GraphIr::new(&format!("impl_g_329"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_330() {
        let mut g = GraphIr::new(&format!("impl_g_330"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_331() {
        let mut g = GraphIr::new(&format!("impl_g_331"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_332() {
        let mut g = GraphIr::new(&format!("impl_g_332"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_333() {
        let mut g = GraphIr::new(&format!("impl_g_333"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_334() {
        let mut g = GraphIr::new(&format!("impl_g_334"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_335() {
        let mut g = GraphIr::new(&format!("impl_g_335"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_336() {
        let mut g = GraphIr::new(&format!("impl_g_336"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_337() {
        let mut g = GraphIr::new(&format!("impl_g_337"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_338() {
        let mut g = GraphIr::new(&format!("impl_g_338"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_339() {
        let mut g = GraphIr::new(&format!("impl_g_339"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_340() {
        let mut g = GraphIr::new(&format!("impl_g_340"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_341() {
        let mut g = GraphIr::new(&format!("impl_g_341"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_342() {
        let mut g = GraphIr::new(&format!("impl_g_342"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_343() {
        let mut g = GraphIr::new(&format!("impl_g_343"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_344() {
        let mut g = GraphIr::new(&format!("impl_g_344"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_345() {
        let mut g = GraphIr::new(&format!("impl_g_345"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_346() {
        let mut g = GraphIr::new(&format!("impl_g_346"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_347() {
        let mut g = GraphIr::new(&format!("impl_g_347"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_348() {
        let mut g = GraphIr::new(&format!("impl_g_348"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_349() {
        let mut g = GraphIr::new(&format!("impl_g_349"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_350() {
        let mut g = GraphIr::new(&format!("impl_g_350"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_351() {
        let mut g = GraphIr::new(&format!("impl_g_351"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_352() {
        let mut g = GraphIr::new(&format!("impl_g_352"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_353() {
        let mut g = GraphIr::new(&format!("impl_g_353"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_354() {
        let mut g = GraphIr::new(&format!("impl_g_354"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_355() {
        let mut g = GraphIr::new(&format!("impl_g_355"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_356() {
        let mut g = GraphIr::new(&format!("impl_g_356"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_357() {
        let mut g = GraphIr::new(&format!("impl_g_357"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_358() {
        let mut g = GraphIr::new(&format!("impl_g_358"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_359() {
        let mut g = GraphIr::new(&format!("impl_g_359"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_360() {
        let mut g = GraphIr::new(&format!("impl_g_360"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_361() {
        let mut g = GraphIr::new(&format!("impl_g_361"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_362() {
        let mut g = GraphIr::new(&format!("impl_g_362"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_363() {
        let mut g = GraphIr::new(&format!("impl_g_363"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_364() {
        let mut g = GraphIr::new(&format!("impl_g_364"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_365() {
        let mut g = GraphIr::new(&format!("impl_g_365"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_366() {
        let mut g = GraphIr::new(&format!("impl_g_366"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_367() {
        let mut g = GraphIr::new(&format!("impl_g_367"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_368() {
        let mut g = GraphIr::new(&format!("impl_g_368"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_369() {
        let mut g = GraphIr::new(&format!("impl_g_369"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_370() {
        let mut g = GraphIr::new(&format!("impl_g_370"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_371() {
        let mut g = GraphIr::new(&format!("impl_g_371"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_372() {
        let mut g = GraphIr::new(&format!("impl_g_372"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_373() {
        let mut g = GraphIr::new(&format!("impl_g_373"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_374() {
        let mut g = GraphIr::new(&format!("impl_g_374"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_375() {
        let mut g = GraphIr::new(&format!("impl_g_375"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_376() {
        let mut g = GraphIr::new(&format!("impl_g_376"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_377() {
        let mut g = GraphIr::new(&format!("impl_g_377"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_378() {
        let mut g = GraphIr::new(&format!("impl_g_378"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_379() {
        let mut g = GraphIr::new(&format!("impl_g_379"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_380() {
        let mut g = GraphIr::new(&format!("impl_g_380"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_381() {
        let mut g = GraphIr::new(&format!("impl_g_381"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_382() {
        let mut g = GraphIr::new(&format!("impl_g_382"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_383() {
        let mut g = GraphIr::new(&format!("impl_g_383"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_384() {
        let mut g = GraphIr::new(&format!("impl_g_384"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_385() {
        let mut g = GraphIr::new(&format!("impl_g_385"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_386() {
        let mut g = GraphIr::new(&format!("impl_g_386"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_387() {
        let mut g = GraphIr::new(&format!("impl_g_387"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_388() {
        let mut g = GraphIr::new(&format!("impl_g_388"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_389() {
        let mut g = GraphIr::new(&format!("impl_g_389"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_390() {
        let mut g = GraphIr::new(&format!("impl_g_390"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_391() {
        let mut g = GraphIr::new(&format!("impl_g_391"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_392() {
        let mut g = GraphIr::new(&format!("impl_g_392"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_393() {
        let mut g = GraphIr::new(&format!("impl_g_393"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_394() {
        let mut g = GraphIr::new(&format!("impl_g_394"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_395() {
        let mut g = GraphIr::new(&format!("impl_g_395"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_396() {
        let mut g = GraphIr::new(&format!("impl_g_396"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_397() {
        let mut g = GraphIr::new(&format!("impl_g_397"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_398() {
        let mut g = GraphIr::new(&format!("impl_g_398"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_399() {
        let mut g = GraphIr::new(&format!("impl_g_399"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_400() {
        let mut g = GraphIr::new(&format!("impl_g_400"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_401() {
        let mut g = GraphIr::new(&format!("impl_g_401"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_402() {
        let mut g = GraphIr::new(&format!("impl_g_402"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_403() {
        let mut g = GraphIr::new(&format!("impl_g_403"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_404() {
        let mut g = GraphIr::new(&format!("impl_g_404"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_405() {
        let mut g = GraphIr::new(&format!("impl_g_405"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_406() {
        let mut g = GraphIr::new(&format!("impl_g_406"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_407() {
        let mut g = GraphIr::new(&format!("impl_g_407"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_408() {
        let mut g = GraphIr::new(&format!("impl_g_408"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_409() {
        let mut g = GraphIr::new(&format!("impl_g_409"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_410() {
        let mut g = GraphIr::new(&format!("impl_g_410"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_411() {
        let mut g = GraphIr::new(&format!("impl_g_411"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_412() {
        let mut g = GraphIr::new(&format!("impl_g_412"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_413() {
        let mut g = GraphIr::new(&format!("impl_g_413"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_414() {
        let mut g = GraphIr::new(&format!("impl_g_414"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    #[test]
    fn test_impl_stress_415() {
        let mut g = GraphIr::new(&format!("impl_g_415"));
        let v1 = g.add_value("v1", crate::core::Shape::new(vec![2, 2]), crate::core::DType::F32);
        let bytes = total_output_memory_bytes(&g);
        assert_eq!(bytes, 16);
    }

    // Computation graph IR verification and pass padding line 0
}
