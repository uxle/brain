//! # Fluent API Recurrent Network Builder
//!
//! Ergonomic builder pattern for configuring and assembling multi-layer / bidirectional models.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown, clippy::module_inception, clippy::manual_memcpy)]

use super::config::{CellKind, RnnConfig};
use super::core::RnnResult;
use super::seq::lstm_seq::LstmSeq;

/// Fluent API Builder for Recurrent Networks.
#[derive(Debug, Clone)]
pub struct RnnBuilder {
    pub config: RnnConfig,
}

impl RnnBuilder {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        Self {
            config: RnnConfig::new(input_dim, hidden_dim),
        }
    }

    pub fn lstm(mut self) -> Self {
        self.config.cell.kind = CellKind::Lstm;
        self
    }

    pub fn gru(mut self) -> Self {
        self.config.cell.kind = CellKind::Gru;
        self
    }

    pub fn num_layers(mut self, layers: usize) -> Self {
        self.config.num_layers = layers;
        self
    }

    pub fn bidirectional(mut self, bi: bool) -> Self {
        self.config.bidirectional = bi;
        self
    }

    pub fn dropout(mut self, p: f64) -> Self {
        self.config.dropout = p;
        self
    }

    pub fn build(self) -> RnnResult<LstmSeq> {
        self.config.validate()?;
        Ok(LstmSeq::new(
            self.config.cell.input_dim,
            self.config.cell.hidden_dim,
            self.config.num_layers,
        ))
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
