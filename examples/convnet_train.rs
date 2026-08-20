//! # End-to-End ConvNet Training Example
//!
//! Demonstrates training a convolutional neural network with Brain.

use brain_core::Tensor;
use brain_train::{Batch, Conv2d, Flatten, Linear, MaxPool2d, ReLU, Sequential, Trainer};

fn main() {
    println!("=== Brain 1.0 ConvNet Training Example ===");

    // Create 8 synthetic 4D images of size [1, 6, 6]
    let mut data_vec = Vec::with_capacity(8 * 36);
    let mut targets = Vec::with_capacity(8);

    // Class 0: top-left bright pattern
    for i in 0..4 {
        let mut img = vec![0.05; 36];
        img[0] = 3.0 + i as f64 * 0.1;
        img[1] = 3.0;
        img[6] = 3.0;
        data_vec.extend(img);
        targets.push(0);
    }
    // Class 1: bottom-right bright pattern
    for i in 0..4 {
        let mut img = vec![0.05; 36];
        img[35] = 3.0 + i as f64 * 0.1;
        img[34] = 3.0;
        img[29] = 3.0;
        data_vec.extend(img);
        targets.push(1);
    }

    let inputs = Tensor::from_vec(data_vec, vec![8, 1, 6, 6]);
    let batch = Batch::new(inputs, targets).expect("Valid batch");
    let batches = vec![batch];

    // Architecture: Conv2d(1->4, 3x3) -> ReLU -> MaxPool2d(2x2) -> Flatten -> Linear(4*3*3 -> 2)
    let model = Sequential::new()
        .add(Conv2d::new(1, 4, 3, true))
        .add(ReLU::new())
        .add(MaxPool2d::new(2, 2))
        .add(Flatten::new())
        .add(Linear::new(4 * 3 * 3, 2, true));

    let mut trainer = Trainer::builder()
        .model(model)
        .learning_rate(0.1)
        .build()
        .expect("Trainer initialized");

    let before = trainer.evaluate(&batches).expect("Initial evaluation");
    println!(
        "Initial metrics: loss={:.4}, accuracy={:.1}%",
        before.loss,
        before.accuracy * 100.0
    );

    let after = trainer.fit(&batches, 20).expect("Training fit");
    println!(
        "Trained metrics: loss={:.4}, accuracy={:.1}%",
        after.loss,
        after.accuracy * 100.0
    );

    println!("ConvNet training completed successfully!");
}
