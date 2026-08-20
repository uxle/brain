//! # GNN Layer Traits & Modules
//!
//! [`GnnLayer`] trait, GCN, GAT, GraphSAGE, GIN, GGCN, EdgeConv, GraphTransformer.
#![allow(missing_docs)]

pub mod edge_conv;
pub mod gat;
pub mod gated;
pub mod gcn;
pub mod gin;
pub mod sage;
pub mod transformer;

pub use edge_conv::EdgeConv;
pub use gat::GatLayer;
pub use gated::GatedConv;
pub use gcn::GcnLayer;
pub use gin::GinLayer;
pub use sage::SageLayer;
pub use transformer::GraphTransformerLayer;

use crate::graph::Graph;
use brain_core::Tensor;

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
