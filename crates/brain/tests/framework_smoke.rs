use brain::prelude::*;

#[test]
fn facade_trains_evaluates_and_round_trips_state() {
    let data = SyntheticClassification::two_class_points(6);
    let batches = data.batches(3);
    let model = Sequential::new()
        .add(Linear::new(2, 6, true))
        .add(ReLU::new())
        .add(Linear::new(6, 2, true));

    let mut trainer = Trainer::builder()
        .model(model)
        .learning_rate(0.2)
        .build()
        .unwrap();

    let before = trainer.evaluate(&batches).unwrap();
    let after = trainer.fit(&batches, 6).unwrap();
    let eval = trainer.evaluate(&batches).unwrap();

    assert!(after.loss <= before.loss);
    assert!(eval.accuracy >= 0.5);

    let state = trainer.state();
    let decoded = ModelState::from_brain_bytes(&state.to_brain_bytes()).unwrap();
    assert_eq!(decoded.tensors.len(), state.tensors.len());
}

#[test]
fn tensor_and_autograd_bridge_smoke() {
    let tensor = Tensor::from_vec(vec![3.0], vec![1]);
    let value = tensor_to_value(&tensor, true);
    assert_eq!(value_to_tensor(&value).get(0), 3.0);
}
