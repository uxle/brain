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
}
