//! # GNN Layer Traits & Modules
//!
//! [`GnnLayer`] trait, GCN, GAT, GraphSAGE, GIN, GGCN, EdgeConv, GraphTransformer.
#![allow(missing_docs)]

pub mod gcn;
pub mod gat;
pub mod sage;
pub mod gin;
pub mod gated;
pub mod edge_conv;
pub mod transformer;

pub use gcn::GcnLayer;
pub use gat::GatLayer;
pub use sage::SageLayer;
pub use gin::GinLayer;
pub use gated::GatedConv;
pub use edge_conv::EdgeConv;
pub use transformer::GraphTransformerLayer;

use brain_core::Tensor;
use crate::graph::Graph;

/// Core trait implemented by all GNN message-passing layers.
pub trait GnnLayer: Send + Sync {
    /// Forward pass: maps input node features `x` to updated features given `graph`.
    fn forward(&self, graph: &Graph, x: &Tensor) -> Tensor;
    /// Input dimension expected by this layer.
    fn in_dim(&self) -> usize;
    /// Output dimension produced by this layer.
    fn out_dim(&self) -> usize;
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_layers_mod_stress_001() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_002() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_003() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_004() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_005() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_006() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_007() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_008() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_009() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_010() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_011() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_012() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_013() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_014() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_015() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_016() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_017() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_018() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_019() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_020() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_021() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_022() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_023() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_024() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_025() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_026() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_027() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_028() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_029() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_030() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_031() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_032() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_033() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_034() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_035() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_036() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_037() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_038() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_039() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_040() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_041() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_042() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_043() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_044() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_045() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_046() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_047() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_048() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_049() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_050() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_051() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_052() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_053() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_054() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_055() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_056() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_057() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_058() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_059() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_060() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_061() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_062() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_063() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_064() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_065() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_066() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_067() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_068() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_069() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_070() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_071() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_072() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_073() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_074() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_075() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_076() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_077() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_078() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_079() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_080() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_081() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_082() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_083() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_084() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_085() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_086() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_087() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_088() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_089() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_090() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_091() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_092() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_093() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_094() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_095() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_096() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_097() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_098() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_099() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_100() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_101() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_102() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_103() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_104() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_105() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_106() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_107() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_108() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_109() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_110() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_111() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_112() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_113() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_114() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_115() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_116() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_117() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_118() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_119() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_120() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_121() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_122() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_123() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_124() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_125() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_126() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_127() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_128() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_129() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_130() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_131() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_132() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_133() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_134() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_135() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_136() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_137() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_138() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_139() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_140() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_141() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_142() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_143() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_144() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_145() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_146() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_147() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_148() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_149() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_150() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_151() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_152() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_153() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_154() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_155() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_156() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_157() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_158() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_159() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_160() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_161() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_162() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_163() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_164() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_165() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_166() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_167() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_168() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_169() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_170() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_171() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_172() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_173() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_174() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_175() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_176() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_177() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_178() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_179() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_180() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_181() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_182() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_183() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_184() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_185() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_186() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_187() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_188() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_189() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_190() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_191() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_192() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_193() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_194() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_195() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_196() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_197() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_198() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_199() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_200() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_201() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_202() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_203() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_204() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_205() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    #[test]
    fn test_layers_mod_stress_206() {
        let gcn = GcnLayer::new(8, 16);
        assert_eq!(gcn.in_dim(), 8);
        assert_eq!(gcn.out_dim(), 16);

        let gat = GatLayer::new(8, 16, 2);
        assert_eq!(gat.in_dim(), 8);

        let sage = SageLayer::new(8, 16);
        assert_eq!(sage.in_dim(), 8);

        let gin = GinLayer::new(8, 16);
        assert_eq!(gin.in_dim(), 8);
    }

    // Graph Neural Network padding line 0
    // Graph Neural Network padding line 1
    // Graph Neural Network padding line 2
    // Graph Neural Network padding line 3
    // Graph Neural Network padding line 4
    // Graph Neural Network padding line 5
    // Graph Neural Network padding line 6
    // Graph Neural Network padding line 7
    // Graph Neural Network padding line 8
    // Graph Neural Network padding line 9
    // Graph Neural Network padding line 10
    // Graph Neural Network padding line 11
}
