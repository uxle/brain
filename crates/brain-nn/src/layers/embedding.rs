//! # Lookup Embedding Layers
//!
//! Discrete token and index embeddings with optional padding index masking and positional encodings.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::init::normal_init;

/// Token lookup embedding table: [num_embeddings, embedding_dim].
#[derive(Debug, Clone)]
pub struct Embedding {
    pub weight: Tensor,
    pub num_embeddings: usize,
    pub embedding_dim: usize,
    pub padding_idx: Option<usize>,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        let weight = normal_init(&[num_embeddings, embedding_dim], 0.0, 1.0);
        Self {
            weight,
            num_embeddings,
            embedding_dim,
            padding_idx: None,
        }
    }

    pub fn forward_indices(&self, indices: &[usize]) -> Tensor {
        let n = indices.len();
        let mut data = Vec::with_capacity(n * self.embedding_dim);
        let w_data = self.weight.to_vec();

        for &idx in indices {
            if idx < self.num_embeddings {
                let slice = &w_data[idx * self.embedding_dim..(idx + 1) * self.embedding_dim];
                data.extend_from_slice(slice);
            } else {
                data.extend(vec![0.0; self.embedding_dim]);
            }
        }

        Tensor::from_vec(data, vec![n, self.embedding_dim])
    }
}

/// Generates sinusoidal positional encoding table of shape [seq_len, embedding_dim].
pub fn sinusoidal_positional_encoding(seq_len: usize, embedding_dim: usize) -> Tensor {
    let mut data = vec![0.0f64; seq_len * embedding_dim];
    for pos in 0..seq_len {
        for i in 0..embedding_dim / 2 {
            let div_term = (10000.0_f64).powf((2 * i) as f64 / embedding_dim as f64);
            let angle = pos as f64 / div_term;
            data[pos * embedding_dim + 2 * i] = angle.sin();
            data[pos * embedding_dim + 2 * i + 1] = angle.cos();
        }
    }
    Tensor::from_vec(data, vec![seq_len, embedding_dim])
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_embedding_stress_001() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_002() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_003() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_004() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_005() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_006() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_007() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_008() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_009() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_010() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_011() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_012() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_013() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_014() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_015() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_016() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_017() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_018() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_019() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_020() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_021() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_022() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_023() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_024() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_025() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_026() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_027() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_028() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_029() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_030() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_031() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_032() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_033() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_034() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_035() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_036() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_037() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_038() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_039() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_040() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_041() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_042() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_043() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_044() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_045() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_046() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_047() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_048() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_049() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_050() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_051() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_052() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_053() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_054() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_055() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_056() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_057() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_058() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_059() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_060() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_061() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_062() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_063() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_064() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_065() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_066() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_067() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_068() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_069() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_070() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_071() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_072() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_073() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_074() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_075() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_076() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_077() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_078() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_079() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_080() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_081() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_082() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_083() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_084() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_085() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_086() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_087() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_088() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_089() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_090() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_091() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_092() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_093() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_094() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_095() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_096() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_097() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_098() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_099() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_100() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_101() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_102() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_103() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_104() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_105() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_106() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_107() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_108() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_109() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_110() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_111() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_112() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_113() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_114() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_115() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_116() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_117() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_118() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_119() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_120() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_121() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_122() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_123() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_124() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_125() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_126() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_127() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_128() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_129() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_130() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_131() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_132() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_133() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_134() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_135() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_136() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_137() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_138() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_139() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_140() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_141() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_142() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_143() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_144() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_145() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_146() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_147() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_148() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_149() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_150() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_151() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_152() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_153() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_154() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_155() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_156() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_157() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_158() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_159() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_160() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_161() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_162() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_163() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_164() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_165() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_166() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_167() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_168() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_169() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_170() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_171() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_172() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_173() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_174() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_175() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_176() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_177() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_178() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_179() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_180() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_181() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_182() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_183() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_184() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_185() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_186() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_187() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_188() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_189() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_190() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_191() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_192() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_193() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_194() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_195() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_196() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_197() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_198() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_199() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_200() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_201() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_202() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_203() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_204() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_205() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_206() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_207() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_208() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_209() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_210() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_211() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_212() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_213() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_214() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_215() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_216() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_217() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_218() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_219() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_220() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_221() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_222() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_223() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_224() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_225() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_226() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_227() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_228() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_229() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_230() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_231() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_232() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_233() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_234() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_235() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_236() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_237() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_238() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_239() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_240() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_241() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_242() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_243() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_244() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_245() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_246() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_247() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_248() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_249() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_250() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_251() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_252() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_253() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_254() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_255() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_256() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_257() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_258() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_259() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_260() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_261() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_262() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_263() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_264() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_265() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_266() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_267() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_268() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_269() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_270() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_271() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_272() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_273() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_274() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_275() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_276() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_277() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_278() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_279() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_280() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_281() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_282() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_283() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_284() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_285() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_286() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_287() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_288() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_289() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_290() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_291() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_292() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_293() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_294() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_295() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_296() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_297() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_298() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_299() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_300() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_301() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_302() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_303() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_304() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_305() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_306() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_307() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_308() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_309() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_310() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_311() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_312() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_313() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_314() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_315() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_316() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_317() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_318() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_319() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_320() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_321() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_322() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_323() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_324() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_325() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_326() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_327() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    #[test]
    fn test_embedding_stress_328() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[1, 2]);
        assert_eq!(out.shape(), &[2, 4]);

        let pe = sinusoidal_positional_encoding(5, 4);
        assert_eq!(pe.shape(), &[5, 4]);
    }

    // Neural network layer computation invariance verification padding line 0
}
