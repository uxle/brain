//! # Configuration Specs for Recurrent Networks
//!
//! Architectural hyperparameters for cell types, stacked layers, dropout, and bidirectional flags.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown, clippy::module_inception, clippy::manual_memcpy)]

use super::core::{RnnError, RnnResult};

/// Available recurrent cell primitives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CellKind {
    #[default]
    Lstm,
    Gru,
    VanillaRnn,
    PeepholeLstm,
    LayerNormLstm,
}

/// Configuration options for individual recurrent cells.
#[derive(Debug, Clone, PartialEq)]
pub struct CellConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub bias: bool,
    pub kind: CellKind,
}

impl Default for CellConfig {
    fn default() -> Self {
        Self {
            input_dim: 64,
            hidden_dim: 128,
            bias: true,
            kind: CellKind::Lstm,
        }
    }
}

/// Multi-layer sequence-level RNN configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct RnnConfig {
    pub cell: CellConfig,
    pub num_layers: usize,
    pub dropout: f64,
    pub bidirectional: bool,
    pub batch_first: bool,
}

impl Default for RnnConfig {
    fn default() -> Self {
        Self {
            cell: CellConfig::default(),
            num_layers: 1,
            dropout: 0.0,
            bidirectional: false,
            batch_first: true,
        }
    }
}

impl RnnConfig {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        Self {
            cell: CellConfig {
                input_dim,
                hidden_dim,
                bias: true,
                kind: CellKind::Lstm,
            },
            num_layers: 1,
            dropout: 0.0,
            bidirectional: false,
            batch_first: true,
        }
    }

    pub fn validate(&self) -> RnnResult<()> {
        if self.cell.input_dim == 0 {
            return Err(RnnError::InvalidConfig("input_dim must be > 0".into()));
        }
        if self.cell.hidden_dim == 0 {
            return Err(RnnError::InvalidConfig("hidden_dim must be > 0".into()));
        }
        if self.num_layers == 0 {
            return Err(RnnError::InvalidConfig("num_layers must be > 0".into()));
        }
        if !(0.0..=1.0).contains(&self.dropout) {
            return Err(RnnError::InvalidConfig("dropout must be in [0, 1]".into()));
        }
        Ok(())
    }

    pub fn summary(&self) -> String {
        format!(
            "RnnConfig: kind={:?}, input={}, hidden={}, layers={}, bi={}, dropout={}",
            self.cell.kind, self.cell.input_dim, self.cell.hidden_dim, self.num_layers, self.bidirectional, self.dropout
        )
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown, clippy::excessive_precision)]
    use super::*;
    use crate::core::*;
    use crate::config::*;
    use crate::utils::*;
    use crate::ops::*;
    use crate::cells::*;
    use crate::seq::*;
    use crate::init_rnn::*;
    use crate::reg_ops::*;
    use crate::process::*;
    use crate::backward_ops::*;
    use crate::builder::*;
    use crate::helper::*;
    use crate::VERSION;
    use brain_core::Tensor;
}
