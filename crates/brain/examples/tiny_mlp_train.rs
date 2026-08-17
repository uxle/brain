use brain::prelude::*;

fn main() -> TrainResult<()> {
    let data = SyntheticClassification::two_class_points(16);
    let batches = data.batches(8);

    let model = Sequential::new()
        .add(Linear::new(2, 8, true))
        .add(ReLU::new())
        .add(Linear::new(8, 2, true));

    let mut trainer = Trainer::builder()
        .model(model)
        .loss(CrossEntropyLoss::default())
        .learning_rate(0.2)
        .regularizer(L2Regularization::new(1e-5))
        .build()?;

    let before = trainer.evaluate(&batches)?;
    let after = trainer.fit(&batches, 12)?;
    let eval = trainer.evaluate(&batches)?;

    println!(
        "tiny_mlp_train: before_loss={:.4} train_loss={:.4} eval_accuracy={:.3} steps={}",
        before.loss, after.loss, eval.accuracy, after.steps
    );

    Ok(())
}
