use brain_core::{BrainModelFile, NodeCoord3D, Tensor, BN_MAGIC};

#[test]
fn test_brain_binary_container_roundtrip() {
    let mut model = BrainModelFile::new("vision_transformer_3d")
        .with_meta("framework", "brain")
        .with_meta("spatial_layout", "cubic_mesh_10x10x10")
        .with_meta("total_neurons", "1000000");

    let q_weight = Tensor::from_slice(&[0.1, 0.2, 0.3, 0.4], vec![2, 2]);
    let k_weight = Tensor::from_slice(&[0.5, 0.6, 0.7, 0.8], vec![2, 2]);
    let v_weight = Tensor::from_slice(&[0.9, 1.0, 1.1, 1.2], vec![2, 2]);

    model.add_tensor("encoder.attn.q", q_weight, Some(NodeCoord3D::new(0, 0, 0)));
    model.add_tensor("encoder.attn.k", k_weight, Some(NodeCoord3D::new(0, 1, 0)));
    model.add_tensor("encoder.attn.v", v_weight, Some(NodeCoord3D::new(0, 2, 0)));

    let bytes = model.to_bytes().expect("serialize .bn");
    assert!(bytes.starts_with(BN_MAGIC));

    let loaded = BrainModelFile::from_bytes(&bytes).expect("deserialize .bn");
    assert_eq!(loaded.name, "vision_transformer_3d");
    assert_eq!(loaded.metadata.get("framework").unwrap(), "brain");
    assert_eq!(
        loaded.metadata.get("spatial_layout").unwrap(),
        "cubic_mesh_10x10x10"
    );

    assert_eq!(
        loaded.node_coords.get("encoder.attn.q").unwrap(),
        &NodeCoord3D::new(0, 0, 0)
    );
    assert_eq!(
        loaded.node_coords.get("encoder.attn.k").unwrap(),
        &NodeCoord3D::new(0, 1, 0)
    );
    assert_eq!(
        loaded.node_coords.get("encoder.attn.v").unwrap(),
        &NodeCoord3D::new(0, 2, 0)
    );

    let q = loaded.archive.get("encoder.attn.q").unwrap();
    assert_eq!(q.data(), &[0.1, 0.2, 0.3, 0.4]);
}

#[test]
fn test_bn_crc_tamper_detection() {
    let mut model = BrainModelFile::new("secure_brain");
    model.add_tensor(
        "w",
        Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]),
        None,
    );

    let mut bytes = model.to_bytes().expect("serialize");
    let corrupt_idx = bytes.len() / 2;
    bytes[corrupt_idx] ^= 0xAA; // corrupt payload byte

    let result = BrainModelFile::from_bytes(&bytes);
    assert!(
        result.is_err(),
        "Tampered .bn file must fail CRC validation"
    );
}
