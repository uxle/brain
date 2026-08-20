//! Tests for Federated Learning and FedAvg
use brain_federated::*;
use brain_core::Tensor;

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
