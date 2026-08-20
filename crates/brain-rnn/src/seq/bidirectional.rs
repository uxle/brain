//! # Bidirectional Recurrent Networks
//!
//! Forward and reverse sequence unrolling combined via Concatenation, Summation, or Average.
#![allow(
    missing_docs,
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_is_multiple_of,
    clippy::manual_div_ceil,
    clippy::doc_markdown,
    clippy::module_inception,
    clippy::manual_memcpy
)]

use super::super::core::{RnnResult, RnnState, SequenceOutput};
use super::lstm_seq::LstmSeq;
use super::RnnSequence;
use brain_core::Tensor;

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
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
        merge_mode: BidirectionalMerge,
    ) -> Self {
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
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant,
        clippy::needless_range_loop,
        clippy::manual_div_ceil,
        clippy::manual_is_multiple_of,
        clippy::too_many_arguments,
        clippy::doc_markdown,
        clippy::excessive_precision
    )]
    use super::*;
    use crate::backward_ops::*;
    use crate::builder::*;
    use crate::cells::*;
    use crate::config::*;
    use crate::core::*;
    use crate::helper::*;
    use crate::init_rnn::*;
    use crate::ops::*;
    use crate::process::*;
    use crate::reg_ops::*;
    use crate::seq::*;
    use crate::utils::*;
    use crate::VERSION;
    use brain_core::Tensor;
}
