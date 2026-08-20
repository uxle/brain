//! # GNN Model Builder
//!
//! Fluent builder API for assembling stacked GNN models.
#![allow(missing_docs)]

use crate::config::{GnnConfig, LayerType, PoolingType};

/// Builder for constructing a `GnnConfig`.
#[derive(Debug, Default)]
pub struct GnnBuilder {
    config: GnnConfig,
}

impl GnnBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn layer_type(mut self, layer_type: LayerType) -> Self {
        self.config.layer_type = layer_type;
        self
    }

    pub fn hidden_dim(mut self, dim: usize) -> Self {
        self.config.hidden_dim = dim;
        self
    }

    pub fn num_layers(mut self, n: usize) -> Self {
        self.config.num_layers = n;
        self
    }

    pub fn num_classes(mut self, c: usize) -> Self {
        self.config.num_classes = c;
        self
    }

    pub fn pooling(mut self, p: PoolingType) -> Self {
        self.config.pooling = p;
        self
    }

    pub fn dropout(mut self, d: f64) -> Self {
        self.config.dropout = d;
        self
    }

    pub fn use_residual(mut self, res: bool) -> Self {
        self.config.use_residual = res;
        self
    }

    pub fn build(self) -> Result<GnnConfig, String> {
        self.config.validate()?;
        Ok(self.config)
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
