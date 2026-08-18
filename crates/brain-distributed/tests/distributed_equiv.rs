//! # Distributed Equivalence Verification Tests

use brain_core::Tensor;
use brain_distributed::data_parallel::DataParallel;

#[test]
fn test_distributed_2rank_allreduce_equivalence() {
    let dp = DataParallel::new(2);

    // Rank 0 gradients: [0.2, 0.4]
    let g0 = Tensor::from_vec(vec![0.2, 0.4], vec![2]);
    // Rank 1 gradients: [0.6, 0.8]
    let g1 = Tensor::from_vec(vec![0.6, 0.8], vec![2]);

    // Combined 2-rank average gradient: [(0.2+0.6)/2, (0.4+0.8)/2] = [0.4, 0.6]
    let mut reduced_data = vec![0.0; 2];
    for i in 0..2 {
        reduced_data[i] = (g0.data()[i] + g1.data()[i]) / (dp.world_size as f64);
    }
    let mut reduced = Tensor::from_vec(reduced_data, vec![2]);

    dp.sync_gradients(std::slice::from_mut(&mut reduced));

    assert_eq!(reduced.shape(), &[2]);
    assert!((reduced.data()[0] - 0.4).abs() < 1e-6);
    assert!((reduced.data()[1] - 0.6).abs() < 1e-6);
}
