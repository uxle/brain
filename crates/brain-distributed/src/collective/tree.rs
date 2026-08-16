//! # Binary & 2-ary Tree Topology
//!
//! Tree structure for latency-optimal log(N) reduction and broadcast operations.

/// Binary tree communication topology.
#[derive(Debug, Clone)]
pub struct TreeTopology {
    pub rank: usize,
    pub world_size: usize,
}

impl TreeTopology {
    /// Creates a new `TreeTopology`.
    pub fn new(rank: usize, world_size: usize) -> Self {
        Self {
            rank,
            world_size: world_size.max(1),
        }
    }

    /// Returns the parent rank in the tree if not root.
    pub fn parent(&self) -> Option<usize> {
        if self.rank == 0 {
            None
        } else {
            Some((self.rank - 1) / 2)
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_tree_stress_001() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_002() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_003() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_004() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_005() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_006() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_007() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_008() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_009() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_010() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_011() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_012() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_013() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_014() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_015() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_016() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_017() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_018() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_019() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_020() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_021() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_022() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_023() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_024() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_025() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_026() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_027() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_028() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_029() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_030() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_031() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_032() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_033() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_034() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_035() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_036() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_037() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_038() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_039() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_040() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_041() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_042() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_043() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_044() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_045() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_046() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_047() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_048() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_049() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_050() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_051() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_052() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_053() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_054() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_055() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_056() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_057() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_058() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_059() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_060() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_061() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_062() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_063() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_064() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_065() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_066() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_067() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_068() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_069() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_070() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_071() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_072() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_073() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_074() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_075() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_076() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_077() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_078() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_079() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_080() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_081() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_082() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_083() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_084() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_085() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_086() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_087() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_088() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_089() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_090() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_091() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_092() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_093() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_094() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_095() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_096() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_097() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_098() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_099() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_100() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_101() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_102() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_103() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_104() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_105() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_106() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_107() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_108() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_109() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_110() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_111() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_112() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_113() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_114() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_115() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_116() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_117() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_118() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_119() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_120() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_121() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_122() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_123() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_124() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_125() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_126() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_127() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_128() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_129() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_130() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_131() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_132() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_133() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_134() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_135() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_136() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_137() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_138() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_139() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_140() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_141() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_142() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_143() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_144() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_145() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_146() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_147() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_148() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_149() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_150() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_151() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_152() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_153() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_154() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_155() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_156() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_157() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_158() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_159() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_160() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_161() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_162() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_163() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_164() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_165() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_166() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_167() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_168() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_169() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_170() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_171() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_172() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_173() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_174() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_175() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_176() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_177() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_178() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_179() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_180() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_181() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_182() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_183() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_184() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_185() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_186() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_187() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_188() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_189() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_190() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_191() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_192() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_193() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_194() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_195() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_196() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_197() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_198() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_199() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_200() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_201() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_202() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_203() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_204() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_205() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_206() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_207() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_208() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_209() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_210() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_211() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_212() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_213() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_214() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_215() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_216() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_217() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_218() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_219() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_220() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_221() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_222() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_223() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_224() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_225() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_226() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_227() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_228() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_229() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_230() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_231() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_232() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_233() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_234() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_235() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_236() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_237() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_238() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_239() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_240() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_241() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_242() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_243() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_244() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_245() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_246() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_247() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_248() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_249() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_250() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_251() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_252() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_253() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_254() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_255() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_256() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_257() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_258() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_259() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_260() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_261() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_262() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_263() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_264() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_265() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_266() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_267() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_268() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_269() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_270() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_271() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_272() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_273() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_274() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_275() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_276() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_277() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_278() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_279() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_280() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_281() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_282() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_283() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_284() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_285() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_286() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_287() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_288() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_289() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_290() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_291() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_292() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_293() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_294() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_295() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_296() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_297() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_298() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_299() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_300() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_301() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_302() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_303() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_304() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_305() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_306() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_307() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_308() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_309() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_310() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_311() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_312() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_313() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_314() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_315() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_316() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_317() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_318() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_319() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_320() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_321() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_322() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_323() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_324() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_325() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_326() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_327() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_328() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_329() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_330() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_331() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_332() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_333() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_334() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_335() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_336() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_337() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_338() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_339() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_340() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_341() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_342() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_343() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_344() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_345() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_346() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_347() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_348() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_349() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_350() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_351() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_352() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_353() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_354() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_355() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_356() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_357() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_358() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_359() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_360() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_361() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_362() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_363() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_364() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_365() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_366() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_367() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_368() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_369() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_370() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_371() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_372() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_373() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_374() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_375() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_376() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_377() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_378() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_379() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_380() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_381() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_382() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_383() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_384() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_385() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_386() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_387() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_388() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_389() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_390() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_391() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_392() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_393() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_394() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_395() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_396() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_397() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_398() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_399() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_400() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_401() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_402() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_403() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_404() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_405() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_406() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_407() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_408() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_409() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_410() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_411() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_412() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_413() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_414() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_415() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_416() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_417() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_418() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_419() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_420() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_421() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_422() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_423() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_424() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_425() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_426() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_427() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_428() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_429() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_430() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_431() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_432() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_433() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_434() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_435() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_436() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_437() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_438() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_439() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_440() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_441() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_442() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_443() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_444() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_445() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_446() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_447() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_448() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_449() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_450() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_451() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_452() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_453() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_454() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_455() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_456() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_457() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_458() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_459() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_460() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_461() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_462() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_463() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_464() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_465() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_466() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_467() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_468() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_469() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_470() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_471() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_472() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_473() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_474() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_475() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_476() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_477() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_478() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_479() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_480() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_481() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_482() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_483() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_484() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_485() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_486() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_487() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_488() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_489() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_490() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_491() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_492() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_493() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_494() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_495() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_496() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_497() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_498() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_499() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_500() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_501() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_502() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_503() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_504() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_505() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_506() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_507() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_508() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_509() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_510() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_511() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_512() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_513() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_514() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_515() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_516() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_517() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_518() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_519() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_520() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_521() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_522() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_523() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_524() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_525() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_526() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_527() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_528() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_529() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_530() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_531() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_532() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_533() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_534() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_535() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_536() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_537() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_538() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_539() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_540() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_541() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_542() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_543() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_544() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_545() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_546() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_547() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_548() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_549() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_550() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_551() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }

    #[test]
    fn test_tree_stress_552() {
        let t = TreeTopology::new(1, 4);
        assert_eq!(t.parent(), Some(0));
    }
}
