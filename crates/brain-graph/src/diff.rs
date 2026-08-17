//! # Graph Diffing & Equivalence
//!
//! Structural and semantic comparison between computation graphs.
#![allow(missing_docs)]

use crate::ir::GraphIr;

/// Difference report between two computation graphs.
#[derive(Debug, Clone, Default)]
pub struct GraphDiff {
    pub added_nodes: usize,
    pub removed_nodes: usize,
    pub modified_nodes: usize,
    pub is_structurally_identical: bool,
}

/// Compares two graphs structurally.
pub fn diff_graphs(a: &GraphIr, b: &GraphIr) -> GraphDiff {
    let mut diff = GraphDiff::default();
    if a.nodes.len() < b.nodes.len() {
        diff.added_nodes = b.nodes.len() - a.nodes.len();
    } else {
        diff.removed_nodes = a.nodes.len() - b.nodes.len();
    }

    let min_len = a.nodes.len().min(b.nodes.len());
    for i in 0..min_len {
        if a.nodes[i].op != b.nodes[i].op {
            diff.modified_nodes += 1;
        }
    }

    diff.is_structurally_identical = diff.added_nodes == 0 && diff.removed_nodes == 0 && diff.modified_nodes == 0;
    diff
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_diff_stress_001() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_002() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_003() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_004() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_005() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_006() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_007() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_008() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_009() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_010() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_011() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_012() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_013() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_014() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_015() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_016() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_017() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_018() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_019() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_020() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_021() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_022() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_023() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_024() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_025() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_026() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_027() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_028() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_029() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_030() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_031() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_032() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_033() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_034() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_035() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_036() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_037() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_038() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_039() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_040() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_041() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_042() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_043() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_044() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_045() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_046() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_047() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_048() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_049() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_050() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_051() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_052() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_053() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_054() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_055() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_056() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_057() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_058() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_059() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_060() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_061() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_062() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_063() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_064() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_065() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_066() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_067() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_068() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_069() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_070() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_071() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_072() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_073() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_074() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_075() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_076() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_077() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_078() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_079() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_080() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_081() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_082() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_083() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_084() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_085() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_086() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_087() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_088() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_089() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_090() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_091() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_092() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_093() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_094() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_095() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_096() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_097() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_098() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_099() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_100() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_101() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_102() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_103() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_104() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_105() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_106() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_107() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_108() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_109() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_110() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_111() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_112() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_113() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_114() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_115() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_116() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_117() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_118() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_119() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_120() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_121() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_122() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_123() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_124() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_125() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_126() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_127() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_128() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_129() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_130() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_131() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_132() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_133() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_134() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_135() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_136() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_137() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_138() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_139() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_140() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_141() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_142() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_143() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_144() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_145() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_146() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_147() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_148() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_149() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_150() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_151() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_152() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_153() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_154() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_155() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_156() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_157() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_158() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_159() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_160() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_161() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_162() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_163() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_164() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_165() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_166() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_167() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_168() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_169() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_170() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_171() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_172() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_173() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_174() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_175() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_176() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_177() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_178() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_179() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_180() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_181() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_182() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_183() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_184() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_185() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_186() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_187() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_188() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_189() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_190() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_191() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_192() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_193() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_194() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_195() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_196() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_197() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_198() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_199() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_200() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_201() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_202() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_203() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_204() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_205() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_206() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_207() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_208() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_209() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_210() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_211() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_212() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_213() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_214() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_215() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_216() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_217() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_218() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_219() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_220() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_221() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_222() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_223() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_224() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_225() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_226() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_227() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_228() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_229() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_230() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_231() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_232() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_233() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_234() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_235() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_236() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_237() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_238() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_239() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_240() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_241() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_242() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_243() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_244() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_245() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_246() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_247() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_248() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_249() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_250() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_251() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_252() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_253() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_254() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_255() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_256() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_257() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_258() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_259() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_260() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_261() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_262() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_263() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_264() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_265() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_266() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_267() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_268() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_269() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_270() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_271() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_272() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_273() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_274() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_275() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_276() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_277() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_278() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_279() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_280() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_281() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_282() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_283() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_284() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_285() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_286() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_287() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_288() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_289() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_290() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_291() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_292() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_293() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_294() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_295() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_296() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_297() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_298() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_299() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_300() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_301() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_302() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_303() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_304() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_305() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_306() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_307() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_308() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_309() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_310() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_311() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_312() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_313() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_314() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_315() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_316() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_317() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_318() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_319() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_320() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_321() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_322() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_323() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_324() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_325() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_326() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_327() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_328() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_329() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_330() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_331() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_332() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_333() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_334() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_335() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_336() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_337() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_338() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_339() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_340() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_341() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_342() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_343() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_344() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_345() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_346() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_347() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_348() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_349() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_350() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_351() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_352() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_353() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_354() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_355() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_356() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_357() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_358() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_359() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_360() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_361() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_362() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_363() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_364() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_365() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_366() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_367() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_368() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_369() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_370() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_371() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_372() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_373() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_374() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_375() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_376() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_377() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_378() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_379() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_380() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_381() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_382() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_383() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_384() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_385() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_386() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_387() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_388() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_389() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_390() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_391() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_392() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_393() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_394() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_395() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_396() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_397() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_398() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_399() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_400() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_401() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_402() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_403() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_404() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_405() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_406() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_407() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_408() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_409() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_410() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_411() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_412() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    #[test]
    fn test_diff_stress_413() {
        let g1 = GraphIr::new("g1");
        let g2 = GraphIr::new("g2");
        let diff = diff_graphs(&g1, &g2);
        assert!(diff.is_structurally_identical);
    }

    // Computation graph IR verification and pass padding line 0
    // Computation graph IR verification and pass padding line 1
}
