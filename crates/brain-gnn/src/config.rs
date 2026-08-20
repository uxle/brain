//! # GNN Configuration
//!
//! Configuration for GNN models, layers, and training parameters.
#![allow(missing_docs)]

/// Layer type for GNN architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LayerType {
    #[default]
    Gcn,
    Gat,
    Sage,
    Gin,
    Gated,
    EdgeConv,
    Transformer,
}

/// Aggregation function for neighborhood aggregation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AggregatorType {
    #[default]
    Mean,
    Sum,
    Max,
    Attention,
    Lstm,
}

/// Pooling function for graph-level readout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PoolingType {
    #[default]
    Add,
    Mean,
    Max,
    Attention,
    Set2Set,
}

/// Configuration for a single GNN layer.
#[derive(Debug, Clone)]
pub struct LayerConfig {
    pub layer_type: LayerType,
    pub in_dim: usize,
    pub out_dim: usize,
    pub num_heads: usize,
    pub dropout: f64,
    pub bias: bool,
    pub aggregator: AggregatorType,
    pub concat_heads: bool,
    pub epsilon: f64,
}

impl Default for LayerConfig {
    fn default() -> Self {
        Self {
            layer_type: LayerType::Gcn,
            in_dim: 16,
            out_dim: 16,
            num_heads: 1,
            dropout: 0.0,
            bias: true,
            aggregator: AggregatorType::Mean,
            concat_heads: true,
            epsilon: 0.0,
        }
    }
}

/// Master GNN architecture configuration.
#[derive(Debug, Clone)]
pub struct GnnConfig {
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_classes: usize,
    pub layer_type: LayerType,
    pub pooling: PoolingType,
    pub dropout: f64,
    pub use_residual: bool,
}

impl Default for GnnConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 64,
            num_layers: 3,
            num_classes: 2,
            layer_type: LayerType::Gcn,
            pooling: PoolingType::Mean,
            dropout: 0.1,
            use_residual: false,
        }
    }
}

impl GnnConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_dim == 0 {
            return Err("hidden_dim must be > 0".into());
        }
        if self.num_layers == 0 {
            return Err("num_layers must be > 0".into());
        }
        Ok(())
    }

    pub fn summary(&self) -> String {
        format!(
            "GNN[type={:?} layers={} hidden={} pooling={:?}]",
            self.layer_type, self.num_layers, self.hidden_dim, self.pooling
        )
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
