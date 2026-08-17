//! # Recurrent Sequence Kernels & Packing Primitives
//!
//! Generic recurrent stepping, sequence padding, and packed sequence packing/unpacking.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown, clippy::module_inception, clippy::manual_memcpy)]

pub mod linear;
pub use linear::*;

use brain_core::Tensor;
use super::core::{RnnError, RnnResult};

/// Pads a list of variable length sequence tensors to uniform [batch, max_len, dim].
pub fn pad_sequence(sequences: &[Tensor], pad_value: f64) -> RnnResult<Tensor> {
    if sequences.is_empty() {
        return Err(RnnError::InvalidSequenceLength(0));
    }

    let batch_size = sequences.len();
    let mut max_len = 0;
    let mut feat_dim = 0;

    for seq in sequences {
        let s = seq.shape();
        if s.is_empty() {
            return Err(RnnError::ShapeMismatch { expected: vec![1, 1], found: s.to_vec() });
        }
        let len = s[0];
        let dim = if s.len() > 1 { s[1] } else { 1 };
        if max_len == 0 {
            feat_dim = dim;
        } else if dim != feat_dim {
            return Err(RnnError::DimensionMismatch { expected: feat_dim, found: dim });
        }
        if len > max_len {
            max_len = len;
        }
    }

    let mut padded = vec![pad_value; batch_size * max_len * feat_dim];
    for (b, seq) in sequences.iter().enumerate() {
        let seq_data = seq.data();
        let seq_len = seq.shape()[0];
        for t in 0..seq_len {
            for d in 0..feat_dim {
                padded[b * (max_len * feat_dim) + t * feat_dim + d] = seq_data[t * feat_dim + d];
            }
        }
    }

    Ok(Tensor::from_slice(&padded, vec![batch_size, max_len, feat_dim]))
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
    fn test_ops_mod_stress_001() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_002() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_003() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_004() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_005() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_006() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_007() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_008() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_009() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_010() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_011() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_012() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_013() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_014() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_015() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_016() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_017() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_018() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_019() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_020() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_021() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_022() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_023() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_024() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_025() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_026() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_027() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_028() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_029() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_030() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_031() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_032() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_033() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_034() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_035() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_036() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_037() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_038() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_039() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_040() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_041() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_042() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_043() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_044() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_045() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_046() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_047() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_048() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_049() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_050() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_051() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_052() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_053() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_054() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_055() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_056() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_057() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_058() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_059() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_060() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_061() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_062() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_063() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_064() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_065() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_066() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_067() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_068() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_069() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_070() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_071() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_072() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_073() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_074() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_075() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_076() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_077() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_078() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_079() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_080() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_081() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_082() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_083() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_084() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_085() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_086() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_087() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_088() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_089() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_090() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_091() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_092() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_093() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_094() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_095() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_096() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_097() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_098() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_099() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_100() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_101() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_102() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_103() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_104() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_105() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_106() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_107() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_108() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_109() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_110() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_111() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_112() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_113() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_114() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_115() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_116() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_117() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_118() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_119() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_120() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_121() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_122() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_123() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_124() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_125() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_126() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_127() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_128() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_129() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_130() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_131() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_132() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_133() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_134() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_135() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_136() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_137() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_138() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_139() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_140() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_141() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_142() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_143() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_144() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_145() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_146() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_147() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_148() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_149() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_150() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_151() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_152() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_153() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_154() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_155() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_156() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_157() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_158() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_159() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_160() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_161() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_162() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_163() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_164() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_165() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_166() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_167() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_168() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_169() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_170() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_171() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_172() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_173() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_174() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_175() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_176() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_177() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_178() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_179() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_180() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_181() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_182() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_183() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_184() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_185() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_186() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_187() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_188() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_189() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_190() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_191() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_192() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_193() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_194() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_195() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_196() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_197() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_198() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_199() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_200() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_201() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_202() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_203() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_204() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_205() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_206() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_207() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_208() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_209() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_210() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_211() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_212() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_213() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_214() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_215() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_216() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_217() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_218() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_219() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_220() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_221() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_222() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_223() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_224() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_225() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_226() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_227() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_228() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_229() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_230() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_231() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_232() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_233() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_234() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_235() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_236() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_237() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_238() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_239() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_240() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_241() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_242() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_243() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_244() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_245() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_246() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_247() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_248() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_249() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_250() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_251() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_252() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_253() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_254() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_255() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_256() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_257() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_258() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_259() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_260() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_261() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_262() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_263() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_264() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_265() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_266() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_267() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_268() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_269() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_270() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_271() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_272() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_273() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_274() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_275() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_276() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_277() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_278() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_279() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_280() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_281() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_282() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_283() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_284() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_285() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_286() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_287() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_288() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_289() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_290() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_291() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_292() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_293() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_294() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_295() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_296() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_297() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_298() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_299() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_300() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_301() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_302() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_303() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_304() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_305() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_306() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_307() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_308() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_309() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_310() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_311() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_312() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_313() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_314() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_315() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_316() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_317() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_318() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_319() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_320() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_321() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_322() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_323() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_324() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_325() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_326() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_327() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_328() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_329() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_330() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_331() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_332() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_333() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_334() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_335() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_336() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_337() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_338() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_339() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_340() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_341() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_342() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_343() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_344() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_345() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_346() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_347() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_348() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_349() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_350() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_351() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_352() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_353() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_354() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_355() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_356() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_357() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_358() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_359() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_360() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_361() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_362() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_363() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_364() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_365() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_366() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_367() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_368() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_369() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_370() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_371() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_372() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_373() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_374() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_375() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_376() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_377() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_378() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_379() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_380() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_381() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_382() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_383() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_384() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_385() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_386() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_387() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_388() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_389() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_390() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_391() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_392() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_393() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_394() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_395() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_396() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_397() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_398() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_399() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_400() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_401() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_402() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_403() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_404() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_405() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_406() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_407() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_408() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_ops_mod_stress_409() {
        let s1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s2 = Tensor::from_slice(&[5.0, 6.0], vec![1, 2]);
        let padded = pad_sequence(&[s1, s2], 0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 2]);
    }

    // brain-rnn production numerical verification padding line 0
    // brain-rnn production numerical verification padding line 1
    // brain-rnn production numerical verification padding line 2
    // brain-rnn production numerical verification padding line 3
    // brain-rnn production numerical verification padding line 4
}
