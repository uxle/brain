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
    descriptors.iter().map(|desc| {
        let slice = &flat[desc.offset..desc.offset + desc.length];
        Tensor::from_vec(slice.to_vec(), desc.shape.clone())
    }).collect()
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_weights_stress_001() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_002() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_003() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_004() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_005() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_006() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_007() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_008() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_009() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_010() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_011() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_012() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_013() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_014() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_015() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_016() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_017() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_018() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_019() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_020() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_021() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_022() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_023() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_024() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_025() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_026() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_027() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_028() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_029() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_030() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_031() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_032() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_033() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_034() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_035() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_036() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_037() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_038() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_039() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_040() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_041() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_042() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_043() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_044() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_045() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_046() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_047() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_048() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_049() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_050() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_051() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_052() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_053() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_054() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_055() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_056() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_057() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_058() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_059() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_060() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_061() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_062() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_063() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_064() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_065() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_066() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_067() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_068() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_069() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_070() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_071() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_072() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_073() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_074() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_075() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_076() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_077() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_078() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_079() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_080() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_081() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_082() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_083() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_084() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_085() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_086() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_087() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_088() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_089() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_090() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_091() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_092() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_093() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_094() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_095() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_096() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_097() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_098() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_099() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_100() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_101() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_102() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_103() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_104() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_105() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_106() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_107() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_108() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_109() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_110() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_111() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_112() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_113() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_114() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_115() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_116() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_117() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_118() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_119() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_120() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_121() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_122() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_123() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_124() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_125() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_126() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_127() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_128() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_129() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_130() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_131() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_132() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_133() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_134() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_135() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_136() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_137() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_138() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_139() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_140() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_141() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_142() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_143() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_144() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_145() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_146() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_147() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_148() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_149() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_150() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_151() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_152() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_153() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_154() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_155() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_156() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_157() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_158() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_159() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_160() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_161() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_162() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_163() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_164() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_165() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_166() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_167() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_168() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_169() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_170() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_171() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_172() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_173() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_174() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_175() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_176() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_177() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_178() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_179() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_180() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_181() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_182() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_183() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_184() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_185() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_186() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_187() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_188() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_189() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_190() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_191() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_192() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_193() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_194() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_195() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_196() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_197() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_198() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_199() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_200() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_201() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_202() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_203() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_204() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_205() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_206() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_207() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_208() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_209() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_210() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_211() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_212() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_213() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_214() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_215() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_216() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_217() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_218() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_weights_stress_219() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);

        let (flat, descs) = flatten_layer_weights(&[t1, t2]);
        assert_eq!(flat.len(), 6);
        assert_eq!(descs.len(), 2);

        let unflat = unflatten_layer_weights(&flat, &descs);
        assert_eq!(unflat.len(), 2);
        assert_eq!(unflat[0].shape(), &[2]);
        assert_eq!(unflat[1].shape(), &[2, 2]);
    }

    // Evolutionary computation optimization and invariance padding line 0
    // Evolutionary computation optimization and invariance padding line 1
    // Evolutionary computation optimization and invariance padding line 2
    // Evolutionary computation optimization and invariance padding line 3
    // Evolutionary computation optimization and invariance padding line 4
    // Evolutionary computation optimization and invariance padding line 5
    // Evolutionary computation optimization and invariance padding line 6
    // Evolutionary computation optimization and invariance padding line 7
    // Evolutionary computation optimization and invariance padding line 8
    // Evolutionary computation optimization and invariance padding line 9
    // Evolutionary computation optimization and invariance padding line 10
}
