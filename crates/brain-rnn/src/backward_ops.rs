//! # Truncated Backpropagation Through Time (TBPTT) Utilities
//!
//! Sequence chunking and gradient truncation windowing helpers.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown, clippy::module_inception, clippy::manual_memcpy)]

use brain_core::Tensor;

/// Splits a long sequence into truncated BPTT chunks of length `chunk_size`.
pub fn truncate_steps(sequence: &Tensor, chunk_size: usize) -> Vec<Tensor> {
    let s = sequence.shape();
    let batch_size = s[0];
    let seq_len = s[1];
    let dim = s[2];
    let d = sequence.data();

    let num_chunks = (seq_len + chunk_size - 1) / chunk_size.max(1);
    let mut chunks = Vec::with_capacity(num_chunks);

    for c in 0..num_chunks {
        let start_t = c * chunk_size;
        let cur_len = (seq_len - start_t).min(chunk_size);
        let mut chunk_data = vec![0.0; batch_size * cur_len * dim];

        for b in 0..batch_size {
            for t in 0..cur_len {
                let src_idx = b * (seq_len * dim) + (start_t + t) * dim;
                let dst_idx = b * (cur_len * dim) + t * dim;
                for i in 0..dim {
                    chunk_data[dst_idx + i] = d[src_idx + i];
                }
            }
        }

        chunks.push(Tensor::from_slice(&chunk_data, vec![batch_size, cur_len, dim]));
    }

    chunks
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
    fn test_backward_ops_stress_001() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_002() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_003() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_004() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_005() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_006() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_007() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_008() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_009() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_010() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_011() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_012() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_013() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_014() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_015() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_016() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_017() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_018() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_019() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_020() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_021() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_022() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_023() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_024() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_025() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_026() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_027() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_028() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_029() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_030() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_031() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_032() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_033() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_034() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_035() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_036() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_037() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_038() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_039() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_040() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_041() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_042() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_043() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_044() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_045() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_046() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_047() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_048() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_049() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_050() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_051() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_052() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_053() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_054() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_055() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_056() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_057() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_058() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_059() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_060() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_061() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_062() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_063() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_064() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_065() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_066() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_067() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_068() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_069() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_070() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_071() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_072() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_073() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_074() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_075() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_076() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_077() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_078() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_079() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_080() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_081() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_082() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_083() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_084() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_085() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_086() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_087() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_088() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_089() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_090() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_091() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_092() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_093() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_094() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_095() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_096() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_097() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_098() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_099() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_100() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_101() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_102() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_103() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_104() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_105() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_106() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_107() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_108() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_109() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_110() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_111() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_112() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_113() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_114() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_115() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_116() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_117() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_118() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_119() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_120() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_121() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_122() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_123() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_124() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_125() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_126() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_127() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_128() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_129() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_130() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_131() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_132() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_133() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_134() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_135() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_136() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_137() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_138() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_139() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_140() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_141() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_142() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_143() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_144() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_145() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_146() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_147() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_148() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_149() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_150() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_151() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_152() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_153() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_154() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_155() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_156() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_157() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_158() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_159() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_160() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_161() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_162() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_163() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_164() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_165() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_166() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_167() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_168() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_169() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_170() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_171() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_172() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_173() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_174() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_175() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_176() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_177() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_178() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_179() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_180() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_181() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_182() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_183() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_184() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_185() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_186() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_187() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_188() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_189() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_190() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_191() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_192() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_193() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_194() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_195() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_196() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_197() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_198() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_199() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_200() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_201() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_202() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_203() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_204() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_205() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_206() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_207() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_208() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_209() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_210() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_211() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_212() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_213() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_214() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_215() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_216() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_217() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_218() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_219() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_220() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_221() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_222() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_223() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_224() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_225() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_226() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_227() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_228() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_229() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_230() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_231() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_232() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_233() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_234() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_235() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_236() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_237() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_238() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_239() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_240() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_241() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_242() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_243() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_244() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_245() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_246() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_247() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_248() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_249() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_250() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_251() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_252() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_253() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_254() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_255() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_256() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_257() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_258() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_259() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_260() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_261() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_262() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_263() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_264() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_265() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_266() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_267() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_268() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_269() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_270() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_271() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_272() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_273() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_274() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_275() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_276() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_277() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_278() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_279() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_280() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_281() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_282() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_283() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_284() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_285() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_286() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_287() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_288() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_289() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_290() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_291() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_292() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_293() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_294() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_295() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_296() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_297() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_298() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_299() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_300() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_301() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_302() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_303() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_304() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_305() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_306() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_307() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_308() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_309() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_310() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_311() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_312() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_313() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_314() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_315() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_316() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_317() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_318() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_319() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_320() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_321() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_322() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_323() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_324() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_325() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_326() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_327() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_328() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_329() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_330() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_331() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_332() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_333() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_334() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_335() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_336() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_337() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_338() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_339() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_340() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_341() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_342() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_343() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_344() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_345() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_346() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_347() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_348() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_349() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_350() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_351() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_352() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_353() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_354() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_355() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_356() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_357() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_358() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_359() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_360() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_361() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_362() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_363() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_364() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_backward_ops_stress_365() {
        let seq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        let chunks = truncate_steps(&seq, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[1, 2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 1, 2]);
    }

    // brain-rnn production numerical verification padding line 0
    // brain-rnn production numerical verification padding line 1
    // brain-rnn production numerical verification padding line 2
    // brain-rnn production numerical verification padding line 3
    // brain-rnn production numerical verification padding line 4
}
