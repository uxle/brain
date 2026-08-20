//! # Layer Weight Encoding & Structural Flattening
//!
//! Structural packing of layer weight and bias tensors into continuous 1D parameter vectors.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Metadata describing a specific layer's weight shape and genome slice offset.
#[derive(Debug, Clone)]
pub struct LayerWeightDescriptor {
    pub name: String,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub length: usize,
}

/// Flattens a sequence of neural network parameter tensors into a single vector.
pub fn flatten_layer_weights(tensors: &[Tensor]) -> (Vec<f64>, Vec<LayerWeightDescriptor>) {
    let mut flat = Vec::new();
    let mut descriptors = Vec::with_capacity(tensors.len());

    for (i, t) in tensors.iter().enumerate() {
        let shape = t.shape().to_vec();
        let length: usize = shape.iter().product();
        let offset = flat.len();

        flat.extend_from_slice(&t.to_vec());
        descriptors.push(LayerWeightDescriptor {
            name: format!("layer_{}", i),
            shape,
            offset,
            length,
        });
    }

    (flat, descriptors)
}

/// Reconstructs parameter tensors from a flat genome vector using descriptors.
pub fn unflatten_layer_weights(flat: &[f64], descriptors: &[LayerWeightDescriptor]) -> Vec<Tensor> {
    descriptors
        .iter()
        .map(|desc| {
            let slice = &flat[desc.offset..desc.offset + desc.length];
            Tensor::from_vec(slice.to_vec(), desc.shape.clone())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
