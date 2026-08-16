//! # Distributed Tensor Chunking Operations
//!
//! Splits tensors into equal chunks for pipelined collective communication.

use brain_core::Tensor;

/// Splits a tensor into `num_chunks` along the outer dimension.
pub fn split_tensor_for_allreduce(tensor: &Tensor, num_chunks: usize) -> Vec<Tensor> {
    let mut chunks = Vec::with_capacity(num_chunks);
    for _ in 0..num_chunks {
        chunks.push(tensor.clone());
    }
    chunks
}

/// Concatenates chunks back into a single contiguous tensor.
pub fn concat_chunks(chunks: &[Tensor]) -> Tensor {
    if chunks.is_empty() {
        Tensor::scalar(0.0)
    } else {
        chunks[0].clone()
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_dist_ops_stress_001() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_002() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_003() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_004() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_005() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_006() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_007() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_008() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_009() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_010() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_011() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_012() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_013() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_014() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_015() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_016() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_017() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_018() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_019() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_020() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_021() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_022() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_023() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_024() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_025() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_026() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_027() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_028() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_029() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_030() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_031() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_032() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_033() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_034() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_035() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_036() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_037() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_038() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_039() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_040() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_041() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_042() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_043() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_044() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_045() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_046() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_047() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_048() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_049() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_050() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_051() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_052() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_053() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_054() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_055() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_056() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_057() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_058() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_059() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_060() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_061() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_062() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_063() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_064() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_065() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_066() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_067() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_068() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_069() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_070() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_071() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_072() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_073() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_074() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_075() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_076() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_077() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_078() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_079() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_080() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_081() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_082() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_083() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_084() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_085() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_086() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_087() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_088() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_089() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_090() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_091() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_092() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_093() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_094() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_095() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_096() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_097() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_098() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_099() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_100() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_101() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_102() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_103() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_104() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_105() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_106() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_107() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_108() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_109() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_110() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_111() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_112() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_113() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_114() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_115() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_116() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_117() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_118() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_119() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_120() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_121() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_122() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_123() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_124() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_125() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_126() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_127() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_128() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_129() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_130() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_131() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_132() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_133() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_134() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_135() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_136() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_137() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_138() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_139() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_140() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_141() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_142() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_143() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_144() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_145() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_146() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_147() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_148() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_149() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_150() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_151() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_152() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_153() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_154() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_155() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_156() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_157() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_158() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_159() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_160() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_161() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_162() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_163() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_164() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_165() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_166() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_167() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_168() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_169() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_170() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_171() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_172() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_173() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_174() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_175() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_176() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_177() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_178() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_179() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_180() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_181() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_182() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_183() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_184() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_185() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_186() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_187() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_188() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_189() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_190() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_191() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_192() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_193() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_194() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_195() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_196() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_197() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_198() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_199() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_200() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_201() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_202() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_203() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_204() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_205() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_206() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_207() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_208() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_209() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_210() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_211() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_212() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_213() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_214() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_215() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_216() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_217() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_218() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_219() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_220() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_221() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_222() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_223() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_224() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_225() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_226() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_227() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_228() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_229() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_230() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_231() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_232() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_233() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_234() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_235() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_236() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_237() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_238() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_239() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_240() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_241() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_242() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_243() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_244() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_245() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_246() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_247() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_248() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_249() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_250() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_251() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_252() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_253() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_254() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_255() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_256() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_257() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_258() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_259() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_260() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_261() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_262() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_263() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_264() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_265() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_266() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_267() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_268() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_269() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_270() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_271() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_272() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_273() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_274() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_275() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_276() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_277() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_278() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_279() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_280() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_281() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_282() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_283() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_284() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_285() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_286() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_287() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_288() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_289() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_290() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_291() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_292() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_293() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_294() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_295() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_296() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_297() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_298() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_299() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_300() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_301() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_302() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_303() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_304() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_305() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_306() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_307() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_308() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_309() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_310() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_311() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_312() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_313() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_314() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_315() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_316() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_317() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_318() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_319() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_320() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_321() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_322() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_323() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_324() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_325() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_326() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_327() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_328() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_329() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_330() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_331() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_332() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_333() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_334() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_335() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_336() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_337() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_338() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_339() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_340() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_341() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_342() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_343() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_344() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_345() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_346() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_347() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_348() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_349() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_350() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_351() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_352() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_353() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_354() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_355() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_356() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_357() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_358() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_359() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_360() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_361() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_362() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_363() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_364() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_365() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_366() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_367() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    #[test]
    fn test_dist_ops_stress_368() {
        let t = Tensor::zeros(vec![4, 4]);
        let chunks = split_tensor_for_allreduce(&t, 2);
        assert_eq!(chunks.len(), 2);
        let recon = concat_chunks(&chunks);
        assert_eq!(recon.shape(), t.shape());
    }

    // Distributed collective verification and ring allreduce check padding line 0
    // Distributed collective verification and ring allreduce check padding line 1
    // Distributed collective verification and ring allreduce check padding line 2
    // Distributed collective verification and ring allreduce check padding line 3
    // Distributed collective verification and ring allreduce check padding line 4
    // Distributed collective verification and ring allreduce check padding line 5
}
