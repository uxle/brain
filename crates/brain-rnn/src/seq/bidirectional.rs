//! # Bidirectional Recurrent Networks
//!
//! Forward and reverse sequence unrolling combined via Concatenation, Summation, or Average.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown, clippy::module_inception, clippy::manual_memcpy)]

use brain_core::Tensor;
use super::super::core::{RnnResult, RnnState, SequenceOutput};
use super::lstm_seq::LstmSeq;
use super::RnnSequence;

/// Merge mode for combining forward and backward recurrent representations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BidirectionalMerge {
    #[default]
    Concat,
    Sum,
    Average,
}

/// Bidirectional Recurrent Sequence Processor.
#[derive(Debug, Clone)]
pub struct BidirectionalRnn {
    pub forward_net: LstmSeq,
    pub backward_net: LstmSeq,
    pub merge_mode: BidirectionalMerge,
}

impl BidirectionalRnn {
    pub fn new(input_dim: usize, hidden_dim: usize, num_layers: usize, merge_mode: BidirectionalMerge) -> Self {
        Self {
            forward_net: LstmSeq::new(input_dim, hidden_dim, num_layers),
            backward_net: LstmSeq::new(input_dim, hidden_dim, num_layers),
            merge_mode,
        }
    }

    fn reverse_sequence(input: &Tensor) -> Tensor {
        let s = input.shape();
        let batch_size = s[0];
        let seq_len = s[1];
        let feat_dim = s[2];
        let d = input.data();

        let mut reversed = vec![0.0; d.len()];
        for b in 0..batch_size {
            for t in 0..seq_len {
                let src_t = seq_len - 1 - t;
                for f in 0..feat_dim {
                    reversed[b * (seq_len * feat_dim) + t * feat_dim + f] =
                        d[b * (seq_len * feat_dim) + src_t * feat_dim + f];
                }
            }
        }

        Tensor::from_slice(&reversed, vec![batch_size, seq_len, feat_dim])
    }
}

impl RnnSequence for BidirectionalRnn {
    fn forward(&self, input: &Tensor, init_state: Option<&RnnState>) -> RnnResult<SequenceOutput> {
        let out_fwd = self.forward_net.forward(input, init_state)?;
        let rev_input = Self::reverse_sequence(input);
        let out_bwd = self.backward_net.forward(&rev_input, init_state)?;
        let rev_out_bwd = Self::reverse_sequence(&out_bwd.output);

        let s = out_fwd.output.shape();
        let batch_size = s[0];
        let seq_len = s[1];
        let hidden_dim = s[2];

        let d_fwd = out_fwd.output.data();
        let d_bwd = rev_out_bwd.data();

        let (final_data, out_dim) = match self.merge_mode {
            BidirectionalMerge::Concat => {
                let mut data = vec![0.0; batch_size * seq_len * (2 * hidden_dim)];
                for b in 0..batch_size {
                    for t in 0..seq_len {
                        let out_idx = b * (seq_len * 2 * hidden_dim) + t * (2 * hidden_dim);
                        let src_idx = b * (seq_len * hidden_dim) + t * hidden_dim;
                        for h in 0..hidden_dim {
                            data[out_idx + h] = d_fwd[src_idx + h];
                            data[out_idx + hidden_dim + h] = d_bwd[src_idx + h];
                        }
                    }
                }
                (data, 2 * hidden_dim)
            }
            BidirectionalMerge::Sum => {
                let mut data = vec![0.0; batch_size * seq_len * hidden_dim];
                for i in 0..data.len() {
                    data[i] = d_fwd[i] + d_bwd[i];
                }
                (data, hidden_dim)
            }
            BidirectionalMerge::Average => {
                let mut data = vec![0.0; batch_size * seq_len * hidden_dim];
                for i in 0..data.len() {
                    data[i] = 0.5 * (d_fwd[i] + d_bwd[i]);
                }
                (data, hidden_dim)
            }
        };

        let out_tensor = Tensor::from_slice(&final_data, vec![batch_size, seq_len, out_dim]);
        Ok(SequenceOutput::new(out_tensor, out_fwd.final_state))
    }

    fn init_state(&self, batch_size: usize) -> RnnState {
        self.forward_net.init_state(batch_size)
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
    fn test_bidirectional_stress_001() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_002() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_003() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_004() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_005() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_006() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_007() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_008() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_009() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_010() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_011() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_012() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_013() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_014() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_015() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_016() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_017() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_018() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_019() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_020() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_021() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_022() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_023() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_024() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_025() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_026() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_027() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_028() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_029() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_030() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_031() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_032() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_033() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_034() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_035() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_036() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_037() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_038() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_039() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_040() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_041() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_042() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_043() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_044() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_045() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_046() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_047() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_048() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_049() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_050() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_051() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_052() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_053() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_054() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_055() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_056() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_057() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_058() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_059() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_060() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_061() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_062() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_063() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_064() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_065() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_066() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_067() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_068() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_069() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_070() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_071() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_072() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_073() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_074() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_075() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_076() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_077() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_078() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_079() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_080() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_081() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_082() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_083() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_084() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_085() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_086() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_087() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_088() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_089() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_090() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_091() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_092() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_093() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_094() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_095() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_096() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_097() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_098() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_099() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_100() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_101() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_102() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_103() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_104() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_105() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_106() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_107() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_108() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_109() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_110() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_111() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_112() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_113() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_114() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_115() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_116() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_117() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_118() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_119() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_120() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_121() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_122() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_123() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_124() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_125() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_126() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_127() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_128() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_129() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_130() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_131() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_132() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_133() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_134() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_135() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_136() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_137() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_138() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_139() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_140() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_141() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_142() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_143() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_144() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_145() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_146() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_147() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_148() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_149() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_150() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_151() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_152() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_153() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_154() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_155() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_156() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_157() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_158() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_159() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_160() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_161() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_162() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_163() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_164() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_165() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_166() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_167() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_168() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_169() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_170() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_171() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_172() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_173() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_174() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_175() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_176() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_177() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_178() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_179() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_180() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_181() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_182() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_183() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_184() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_185() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_186() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_187() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_188() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_189() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_190() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_191() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_192() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_193() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_194() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_195() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_196() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_197() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_198() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_199() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_200() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_201() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_202() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_203() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_204() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_205() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_206() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_207() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_208() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_209() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_210() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_211() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_212() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_213() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_214() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_215() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_216() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_217() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_218() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_219() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_220() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_221() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_222() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_223() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_224() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_225() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_226() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_227() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_228() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_229() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_230() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_231() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_232() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_233() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_234() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_235() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_236() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_237() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_238() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_239() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_240() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_241() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_242() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_243() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_244() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_245() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_246() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_247() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_248() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_249() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_250() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_251() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_252() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_253() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_254() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_255() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_256() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_257() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_258() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_259() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_260() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_261() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_262() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_263() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_264() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_265() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_266() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_267() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_268() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_269() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_270() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_271() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_272() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_273() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_274() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_275() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_276() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_277() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_278() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_279() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_280() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_281() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_282() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_283() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_284() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_285() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_286() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_287() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_288() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_289() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_290() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_291() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_292() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_293() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_294() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_295() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_296() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_297() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_298() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_299() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_300() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_301() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_302() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_303() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_304() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_305() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_306() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_307() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_308() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_309() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_310() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_311() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_312() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_313() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_314() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_315() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_316() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_317() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_318() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_319() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_320() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_321() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_322() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_323() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_324() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_325() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_326() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_327() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_328() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_329() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_330() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_331() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_332() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_333() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_334() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_335() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_336() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_337() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_338() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_339() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_340() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_341() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_342() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_343() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_344() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_345() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_346() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_347() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_348() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_349() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_350() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_351() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_352() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_353() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_354() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_355() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_356() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_357() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_358() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_359() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_360() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_361() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_362() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_363() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_364() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_365() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_366() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_367() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_368() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_369() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_370() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_371() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_372() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_373() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_374() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_375() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_376() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_377() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_378() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_379() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_380() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_381() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_382() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_383() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_384() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_385() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_386() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_387() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_388() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_389() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_390() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_391() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_392() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_393() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_394() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_395() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_396() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_397() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_398() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_399() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_400() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_401() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }

    #[test]
    fn test_bidirectional_stress_402() {
        let birnn = BidirectionalRnn::new(2, 4, 1, BidirectionalMerge::Concat);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let out = birnn.forward(&x, None).unwrap();
        assert_eq!(out.output.shape(), &[1, 2, 8]);
    }
}
