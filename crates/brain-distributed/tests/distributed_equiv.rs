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

    // Combined 2-rank sum: [0.2+0.6, 0.4+0.8] = [0.8, 1.2]
    let mut reduced_data = vec![0.0; 2];
    for i in 0..2 {
        reduced_data[i] = g0.data()[i] + g1.data()[i];
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

#[test]
fn test_ring_allreduce_multi_rank_convergence() {
    use brain_distributed::collective::ring_allreduce_simulate;

    let t0 = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]);
    let t1 = Tensor::from_slice(&[4.0, 5.0, 6.0], vec![3]);
    let t2 = Tensor::from_slice(&[7.0, 8.0, 9.0], vec![3]);

    let res = ring_allreduce_simulate(&[t0, t1, t2]);
    assert_eq!(res.len(), 3);

    // Expected sum: [1+4+7, 2+5+8, 3+6+9] = [12.0, 15.0, 18.0]
    for r in res {
        assert_eq!(r.data(), &[12.0, 15.0, 18.0]);
    }
}

#[test]
fn test_data_parallel_ensemble_sync() {
    let r0_params = vec![
        Tensor::from_slice(&[10.0, 20.0], vec![2]),
        Tensor::from_slice(&[1.0], vec![1]),
    ];
    let r1_params = vec![
        Tensor::from_slice(&[30.0, 40.0], vec![2]),
        Tensor::from_slice(&[3.0], vec![1]),
    ];

    let mut ensemble = vec![r0_params, r1_params];
    DataParallel::sync_ensemble_gradients(&mut ensemble);

    assert_eq!(ensemble[0][0].data(), &[20.0, 30.0]);
    assert_eq!(ensemble[0][1].data(), &[2.0]);
    assert_eq!(ensemble[1][0].data(), &[20.0, 30.0]);
    assert_eq!(ensemble[1][1].data(), &[2.0]);
}
