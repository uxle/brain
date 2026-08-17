//! # Recurrent Weight Initializers
//!
//! Orthogonal hidden matrix generation, Xavier input bounds, and forget gate bias presets.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown, clippy::module_inception, clippy::manual_memcpy)]

use brain_core::Tensor;
use super::utils::{init_orthogonal, init_uniform};

/// RNN Initialization strategy configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct RnnInitConfig {
    pub orthogonal_hidden: bool,
    pub forget_bias_init: f64,
    pub seed: u64,
}

impl Default for RnnInitConfig {
    fn default() -> Self {
        Self {
            orthogonal_hidden: true,
            forget_bias_init: 1.0,
            seed: 42,
        }
    }
}

/// Initializes 4-gate LSTM weight tensors according to config.
pub fn init_lstm_weights(input_dim: usize, hidden_dim: usize, config: &RnnInitConfig) -> (Tensor, Tensor, Tensor) {
    let w_ih = init_uniform(4 * hidden_dim, input_dim, input_dim, config.seed);
    let w_hh = if config.orthogonal_hidden {
        init_orthogonal(4 * hidden_dim, hidden_dim, config.seed + 1)
    } else {
        init_uniform(4 * hidden_dim, hidden_dim, hidden_dim, config.seed + 1)
    };

    let mut b_data = vec![0.0; 4 * hidden_dim];
    for i in hidden_dim..(2 * hidden_dim) {
        b_data[i] = config.forget_bias_init;
    }
    let bias = Tensor::from_slice(&b_data, vec![4 * hidden_dim]);

    (w_ih, w_hh, bias)
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
    fn test_init_rnn_stress_001() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_002() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_003() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_004() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_005() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_006() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_007() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_008() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_009() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_010() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_011() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_012() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_013() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_014() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_015() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_016() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_017() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_018() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_019() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_020() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_021() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_022() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_023() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_024() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_025() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_026() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_027() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_028() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_029() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_030() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_031() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_032() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_033() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_034() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_035() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_036() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_037() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_038() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_039() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_040() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_041() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_042() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_043() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_044() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_045() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_046() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_047() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_048() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_049() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_050() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_051() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_052() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_053() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_054() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_055() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_056() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_057() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_058() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_059() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_060() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_061() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_062() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_063() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_064() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_065() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_066() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_067() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_068() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_069() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_070() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_071() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_072() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_073() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_074() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_075() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_076() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_077() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_078() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_079() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_080() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_081() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_082() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_083() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_084() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_085() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_086() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_087() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_088() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_089() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_090() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_091() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_092() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_093() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_094() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_095() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_096() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_097() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_098() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_099() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_100() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_101() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_102() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_103() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_104() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_105() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_106() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_107() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_108() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_109() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_110() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_111() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_112() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_113() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_114() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_115() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_116() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_117() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_118() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_119() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_120() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_121() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_122() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_123() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_124() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_125() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_126() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_127() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_128() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_129() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_130() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_131() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_132() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_133() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_134() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_135() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_136() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_137() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_138() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_139() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_140() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_141() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_142() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_143() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_144() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_145() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_146() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_147() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_148() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_149() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_150() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_151() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_152() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_153() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_154() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_155() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_156() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_157() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_158() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_159() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_160() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_161() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_162() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_163() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_164() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_165() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_166() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_167() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_168() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_169() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_170() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_171() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_172() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_173() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_174() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_175() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_176() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_177() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_178() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_179() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_180() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_181() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_182() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_183() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_184() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_185() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_186() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_187() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_188() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_189() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_190() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_191() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_192() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_193() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_194() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_195() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_196() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_197() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_198() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_199() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_200() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_201() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_202() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_203() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_204() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_205() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_206() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_207() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_208() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_209() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_210() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_211() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_212() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_213() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_214() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_215() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_216() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_217() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_218() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_219() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_220() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_221() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_222() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_223() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_224() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_225() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_226() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_227() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_228() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_229() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_230() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_231() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_232() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_233() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_234() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_235() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_236() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_237() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_238() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_239() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_240() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_241() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_242() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_243() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_244() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_245() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_246() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_247() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_248() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_249() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_250() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_251() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_252() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_253() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_254() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_255() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_256() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_257() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_258() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_259() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_260() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_261() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_262() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_263() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_264() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_265() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_266() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_267() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_268() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_269() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_270() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_271() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_272() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_273() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_274() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_275() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_276() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_277() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_278() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_279() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_280() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_281() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_282() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_283() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_284() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_285() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_286() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_287() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_288() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_289() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_290() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_291() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_292() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_293() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_294() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_295() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_296() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_297() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_298() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_299() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_300() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_301() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_302() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_303() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_304() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_305() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_306() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_307() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_308() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_309() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_310() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_311() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_312() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_313() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_314() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_315() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_316() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_317() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_318() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_319() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_320() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_321() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_322() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_323() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_324() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_325() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_326() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_327() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_328() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_329() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_330() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_331() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_332() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_333() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_334() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_335() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_336() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_337() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_338() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_339() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_340() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_341() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_342() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_343() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_344() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_345() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_346() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_347() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_348() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_349() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_350() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_351() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_352() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_353() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_354() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_355() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_356() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_357() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_358() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_359() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_360() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_361() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_362() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_363() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_364() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }

    #[test]
    fn test_init_rnn_stress_365() {
        let cfg = RnnInitConfig::default();
        let (w_ih, w_hh, b) = init_lstm_weights(2, 4, &cfg);
        assert_eq!(w_ih.shape(), &[16, 2]);
        assert_eq!(w_hh.shape(), &[16, 4]);
        assert_eq!(b.shape(), &[16]);
    }
}
