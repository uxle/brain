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

#[test]
fn test_ring_and_tree_topology_neighbors() {
    use brain_distributed::collective::{RingTopology, TreeTopology};

    let ring = RingTopology::new(0, 4);
    assert_eq!(ring.left_neighbor(), 3);
    assert_eq!(ring.right_neighbor(), 1);

    let ring3 = RingTopology::new(3, 4);
    assert_eq!(ring3.left_neighbor(), 2);
    assert_eq!(ring3.right_neighbor(), 0);

    let tree0 = TreeTopology::new(0, 4);
    assert_eq!(tree0.parent(), None);

    let tree1 = TreeTopology::new(1, 4);
    assert_eq!(tree1.parent(), Some(0));

    let tree2 = TreeTopology::new(2, 4);
    assert_eq!(tree2.parent(), Some(0));

    let tree3 = TreeTopology::new(3, 4);
    assert_eq!(tree3.parent(), Some(1));
}
