//! Tests for GNN message passing and graph structures
use brain_core::Tensor;
use brain_gnn::*;

#[test]
fn test_graph_creation_and_degree_ops() {
    let src = vec![0, 1, 2, 3];
    let dst = vec![1, 2, 3, 0];
    let feats = Tensor::zeros(vec![4, 8]);
    let g = Graph::new(4, src, dst, feats).unwrap();

    assert_eq!(g.num_nodes, 4);
    assert_eq!(g.num_edges(), 4);

    let out_deg = out_degrees(&g);
    assert_eq!(out_deg, vec![1, 1, 1, 1]);
}

#[test]
fn test_gcn_layer_forward() {
    let layer = GcnLayer::new(4, 2);
    let src = vec![0, 1, 0, 1, 2];
    let dst = vec![1, 2, 0, 1, 2];
    let feats = Tensor::from_slice(
        &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        vec![3, 4],
    );

    let g = Graph::new(3, src, dst, feats.clone()).unwrap();
    let out = layer.forward(&g, &feats);
    assert_eq!(out.shape(), &[3, 2]);
}

#[test]
fn test_gat_layer_forward() {
    let layer = GatLayer::new(4, 2, 2);
    let src = vec![0, 1, 0, 1, 2];
    let dst = vec![1, 2, 0, 1, 2];
    let feats = Tensor::from_slice(
        &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        vec![3, 4],
    );

    let g = Graph::new(3, src, dst, feats.clone()).unwrap();
    let out = layer.forward(&g, &feats);
    assert_eq!(out.shape(), &[3, 2]);
}

#[test]
fn test_sage_layer_forward() {
    let layer = SageLayer::new(4, 2);
    let src = vec![0, 1, 2];
    let dst = vec![1, 2, 0];
    let feats = Tensor::from_slice(
        &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        vec![3, 4],
    );

    let g = Graph::new(3, src, dst, feats.clone()).unwrap();
    let out = layer.forward(&g, &feats);
    assert_eq!(out.shape(), &[3, 2]);
}

#[test]
fn test_gin_layer_forward() {
    let layer = GinLayer::new(4, 2);
    let src = vec![0, 1, 2];
    let dst = vec![1, 2, 0];
    let feats = Tensor::from_slice(
        &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        vec![3, 4],
    );

    let g = Graph::new(3, src, dst, feats.clone()).unwrap();
    let out = layer.forward(&g, &feats);
    assert_eq!(out.shape(), &[3, 2]);
}
