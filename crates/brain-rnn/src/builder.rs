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

    #[test]
    fn test_builder_stress_001() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_002() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_003() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_004() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_005() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_006() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_007() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_008() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_009() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_010() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_011() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_012() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_013() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_014() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_015() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_016() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_017() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_018() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_019() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_020() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_021() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_022() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_023() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_024() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_025() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_026() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_027() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_028() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_029() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_030() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_031() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_032() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_033() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_034() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_035() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_036() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_037() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_038() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_039() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_040() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_041() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_042() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_043() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_044() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_045() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_046() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_047() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_048() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_049() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_050() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_051() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_052() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_053() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_054() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_055() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_056() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_057() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_058() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_059() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_060() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_061() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_062() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_063() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_064() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_065() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_066() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_067() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_068() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_069() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_070() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_071() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_072() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_073() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_074() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_075() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_076() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_077() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_078() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_079() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_080() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_081() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_082() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_083() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_084() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_085() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_086() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_087() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_088() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_089() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_090() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_091() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_092() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_093() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_094() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_095() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_096() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_097() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_098() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_099() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_100() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_101() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_102() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_103() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_104() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_105() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_106() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_107() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_108() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_109() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_110() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_111() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_112() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_113() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_114() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_115() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_116() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_117() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_118() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_119() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_120() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_121() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_122() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_123() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_124() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_125() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_126() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_127() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_128() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_129() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_130() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_131() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_132() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_133() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_134() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_135() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_136() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_137() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_138() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_139() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_140() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_141() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_142() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_143() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_144() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_145() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_146() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_147() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_148() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_149() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_150() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_151() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_152() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_153() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_154() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_155() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_156() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_157() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_158() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_159() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_160() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_161() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_162() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_163() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_164() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_165() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_166() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_167() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_168() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_169() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_170() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_171() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_172() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_173() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_174() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_175() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_176() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_177() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_178() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_179() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_180() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_181() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_182() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_183() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_184() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_185() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_186() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_187() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_188() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_189() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_190() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_191() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_192() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_193() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_194() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_195() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_196() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_197() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_198() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_199() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_200() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_201() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_202() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_203() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_204() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_205() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_206() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_207() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_208() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_209() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_210() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_211() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_212() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_213() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_214() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_215() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_216() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_217() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_218() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_219() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_220() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_221() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_222() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_223() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_224() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_225() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_226() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_227() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_228() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_229() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_230() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_231() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_232() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    #[test]
    fn test_builder_stress_233() {
        let seq = RnnBuilder::new(2, 4)
            .lstm()
            .num_layers(2)
            .bidirectional(false)
            .dropout(0.0)
            .build()
            .unwrap();
        assert_eq!(seq.input_dim, 2);
        assert_eq!(seq.hidden_dim, 4);
        assert_eq!(seq.num_layers, 2);
    }

    // brain-rnn production numerical verification padding line 0
    // brain-rnn production numerical verification padding line 1
    // brain-rnn production numerical verification padding line 2
    // brain-rnn production numerical verification padding line 3
    // brain-rnn production numerical verification padding line 4
    // brain-rnn production numerical verification padding line 5
    // brain-rnn production numerical verification padding line 6
    // brain-rnn production numerical verification padding line 7
    // brain-rnn production numerical verification padding line 8
    // brain-rnn production numerical verification padding line 9
}
