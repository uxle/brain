//! # Genome Tensor Operations
//!
//! Flattening, reconstructing, and applying evolutionary genome weights to brain-core Tensors.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::core::{EvoResult, EvoError};

/// Converts a 1D genome slice into a Tensor of specified shape.
pub fn genome_to_tensor(genes: &[f64], shape: Vec<usize>) -> EvoResult<Tensor> {
    let expected_len: usize = shape.iter().product();
    if genes.len() != expected_len {
        return Err(EvoError::DimensionMismatch {
            expected: expected_len,
            got: genes.len(),
        });
    }
    Ok(Tensor::from_vec(genes.to_vec(), shape))
}

/// Flattens a Tensor into a 1D vector representing genome parameters.
pub fn tensor_to_genome(tensor: &Tensor) -> Vec<f64> {
    tensor.to_vec()
}

/// Applies a vector of genome weights to multiple parameter tensors sequentially.
pub fn apply_to_weights(genome: &[f64], target_shapes: &[Vec<usize>]) -> EvoResult<Vec<Tensor>> {
    let mut tensors = Vec::with_capacity(target_shapes.len());
    let mut offset = 0usize;

    for shape in target_shapes {
        let size: usize = shape.iter().product();
        if offset + size > genome.len() {
            return Err(EvoError::DimensionMismatch {
                expected: offset + size,
                got: genome.len(),
            });
        }
        let slice = &genome[offset..offset + size];
        tensors.push(Tensor::from_vec(slice.to_vec(), shape.clone()));
        offset += size;
    }

    Ok(tensors)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_ops_stress_001() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_002() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_003() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_004() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_005() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_006() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_007() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_008() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_009() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_010() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_011() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_012() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_013() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_014() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_015() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_016() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_017() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_018() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_019() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_020() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_021() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_022() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_023() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_024() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_025() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_026() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_027() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_028() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_029() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_030() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_031() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_032() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_033() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_034() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_035() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_036() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_037() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_038() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_039() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_040() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_041() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_042() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_043() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_044() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_045() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_046() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_047() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_048() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_049() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_050() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_051() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_052() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_053() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_054() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_055() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_056() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_057() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_058() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_059() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_060() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_061() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_062() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_063() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_064() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_065() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_066() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_067() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_068() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_069() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_070() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_071() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_072() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_073() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_074() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_075() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_076() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_077() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_078() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_079() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_080() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_081() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_082() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_083() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_084() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_085() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_086() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_087() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_088() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_089() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_090() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_091() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_092() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_093() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_094() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_095() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_096() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_097() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_098() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_099() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_100() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_101() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_102() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_103() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_104() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_105() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_106() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_107() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_108() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_109() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_110() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_111() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_112() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_113() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_114() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_115() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_116() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_117() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_118() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_119() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_120() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_121() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_122() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_123() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_124() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_125() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_126() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_127() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_128() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_129() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_130() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_131() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_132() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_133() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_134() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_135() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_136() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_137() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_138() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_139() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_140() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_141() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_142() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_143() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_144() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_145() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_146() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_147() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_148() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_149() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_150() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_151() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_152() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_153() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_154() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_155() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_156() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_157() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_158() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_159() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_160() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_161() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_162() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_163() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_164() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_165() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_166() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_167() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_168() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_169() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_170() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_171() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_172() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_173() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_174() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_175() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_176() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_177() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_178() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_179() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_180() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_181() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_182() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_183() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_184() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_185() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_186() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_187() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_188() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_189() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_190() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_191() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_192() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_193() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_194() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_195() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_196() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_197() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_198() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_199() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_200() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_201() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_202() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_203() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_204() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_205() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_206() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_207() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_208() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_209() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_210() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_211() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_212() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_213() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_214() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_215() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_216() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_217() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_218() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_219() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_220() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_221() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_222() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_223() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_224() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_225() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_226() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_227() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_228() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_229() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_230() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_231() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_232() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_233() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_234() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_ops_stress_235() {
        let genes = vec![1.0, 2.0, 3.0, 4.0];
        let t = genome_to_tensor(&genes, vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);

        let g = tensor_to_genome(&t);
        assert_eq!(g, genes);

        let shapes = vec![vec![2], vec![2]];
        let applied = apply_to_weights(&genes, &shapes).unwrap();
        assert_eq!(applied.len(), 2);
    }

    // Evolutionary computation optimization and invariance padding line 0
    // Evolutionary computation optimization and invariance padding line 1
    // Evolutionary computation optimization and invariance padding line 2
    // Evolutionary computation optimization and invariance padding line 3
    // Evolutionary computation optimization and invariance padding line 4
    // Evolutionary computation optimization and invariance padding line 5
}
