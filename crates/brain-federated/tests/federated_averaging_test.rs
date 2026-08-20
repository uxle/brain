//! Tests for Federated Learning and FedAvg
use brain_core::Tensor;
use brain_federated::*;

#[test]
fn test_fedavg_aggregation() {
    let t1 = Tensor::from_slice(&[1.0, 2.0], vec![2]);
    let t2 = Tensor::from_slice(&[3.0, 4.0], vec![2]);

    let d1 = ModelDelta::new(1, vec![t1], 10);
    let d2 = ModelDelta::new(2, vec![t2], 10);

    let deltas = vec![d1, d2];
    let avg = fed_avg_aggregate(&deltas);
    assert_eq!(avg.len(), 1);
    assert_eq!(avg[0].to_vec(), vec![2.0, 3.0]);
}

#[test]
fn test_byzantine_robust_trimmed_mean_and_median() {
    let d1 = ModelDelta::new(1, vec![Tensor::from_slice(&[10.0], vec![1])], 10);
    let d2 = ModelDelta::new(2, vec![Tensor::from_slice(&[20.0], vec![1])], 10);
    let d3 = ModelDelta::new(3, vec![Tensor::from_slice(&[30.0], vec![1])], 10);
    let d4 = ModelDelta::new(4, vec![Tensor::from_slice(&[40.0], vec![1])], 10);
    let d_poison = ModelDelta::new(5, vec![Tensor::from_slice(&[99999.0], vec![1])], 10);

    let deltas = vec![d1, d2, d3, d4, d_poison];

    // Median should discard extreme outlier 99999 and return 30.0
    let med = median_aggregate(&deltas);
    assert_eq!(med[0].data()[0], 30.0);

    // Trimmed-mean with 20% trim should drop 10.0 and 99999.0 -> average of [20, 30, 40] = 30.0
    let trim = trimmed_mean_aggregate(&deltas, 0.2);
    assert_eq!(trim[0].data()[0], 30.0);
}

#[test]
fn test_secure_aggregation_protocol() {
    use brain_federated::privacy::{mask_tensor, SecureAggregator};

    let sec_agg = SecureAggregator::new(3);
    let shape = vec![2];
    let seed = 12345;

    let w1 = Tensor::from_slice(&[5.0, 15.0], vec![2]);
    let w2 = Tensor::from_slice(&[25.0, 35.0], vec![2]);
    let w3 = Tensor::from_slice(&[45.0, 55.0], vec![2]);

    let m1 = sec_agg.generate_client_pairwise_mask(0, &shape, seed);
    let m2 = sec_agg.generate_client_pairwise_mask(1, &shape, seed);
    let m3 = sec_agg.generate_client_pairwise_mask(2, &shape, seed);

    let masked1 = mask_tensor(&w1, &m1);
    let masked2 = mask_tensor(&w2, &m2);
    let masked3 = mask_tensor(&w3, &m3);

    let total_server_agg = &(&masked1 + &masked2) + &masked3;
    // True sum: [5+25+45, 15+35+55] = [75.0, 105.0]
    assert!((total_server_agg.data()[0] - 75.0).abs() < 1e-5);
    assert!((total_server_agg.data()[1] - 105.0).abs() < 1e-5);
}
