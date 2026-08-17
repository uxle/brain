//! # `brain make` — build, train, and checkpoint a model.
//!
//! ```text
//! brain make MODEL.brain --data DATA.txt [--hidden H] [--classes C] [--lr F] [--epochs E] [--batch B]
//! ```
//!
//! The `--data` file is a text/CSV dataset whose last column is an integer
//! class label; the preceding columns are the features. `make` builds a
//! `Linear -> ReLU -> Linear` MLP, trains it with SGD on the dataset, then
//! writes a Brain checkpoint (`.brain`) containing the learned weights **and**
//! a compact architecture descriptor so `brain run` can reconstruct the model.

use crate::core::{ExitCode, OutputSink};
use crate::datafile::load;
use crate::parser::ArgParser;

use brain_core::Tensor;
use brain_train::{Batch, Linear, ModelState, ReLU, Sequential, TrainerBuilder};

/// Handles `brain make MODELOUT.brain --data DATA.txt [opts]`.
pub fn run_make_command(args: &[String], sink: &OutputSink) -> ExitCode {
    if args.is_empty() {
        sink.println("Usage: brain make <output.brain> --data <data.txt> [options]");
        sink.println("Options:");
        sink.println("  --hidden N    Hidden units in the MLP (default 16)");
        sink.println("  --classes N   Number of output classes (default: inferred from labels)");
        sink.println("  --lr F        Learning rate (default 0.1)");
        sink.println("  --epochs N    Training epochs (default 20)");
        sink.println("  --batch N     Mini-batch size (default 8)");
        return ExitCode::INVALID_USAGE;
    }

    let out_path = args[0].clone();
    let parser = ArgParser::new()
        .option("data")
        .option("hidden")
        .option("classes")
        .option("lr")
        .option("epochs")
        .option("batch");

    let matches = match parser.parse(&args[1..]) {
        Ok(m) => m,
        Err(err) => {
            sink.println(&format!("error: {}", err));
            return ExitCode::INVALID_USAGE;
        }
    };

    let data_path = match matches.get_option("data") {
        Some(p) => p.to_string(),
        None => {
            sink.println("error: --data <data.txt> is required");
            return ExitCode::INVALID_USAGE;
        }
    };

    let dataset = match load(&data_path, true) {
        Ok(d) => d,
        Err(err) => {
            sink.println(&format!("error: {}", err));
            return ExitCode::ERROR;
        }
    };

    // Determine output classes.
    let n_classes = if let Some(c) = matches.get_option("classes") {
        c.parse::<usize>().unwrap_or_else(|_| {
            sink.println(&format!("error: invalid --classes '{}'", c));
            0
        })
    } else {
        let max_label = dataset.labels.iter().copied().max().unwrap_or(0);
        max_label + 1
    };
    if n_classes == 0 {
        return ExitCode::INVALID_USAGE;
    }

    let hidden: usize = matches
        .get_option("hidden")
        .and_then(|h| h.parse::<usize>().ok())
        .unwrap_or(16);
    let lr: f64 = matches
        .get_option("lr")
        .and_then(|l| l.parse::<f64>().ok())
        .unwrap_or(0.1);
    let epochs: usize = matches
        .get_option("epochs")
        .and_then(|e| e.parse::<usize>().ok())
        .unwrap_or(20);
    let batch_size: usize = matches
        .get_option("batch")
        .and_then(|b| b.parse::<usize>().ok())
        .unwrap_or(8)
        .max(1);

    sink.println(&format!(
        "make: dataset={} samples={} features={} classes={} hidden={} lr={} epochs={} batch={}",
        data_path,
        dataset.features.len(),
        dataset.n_features,
        n_classes,
        hidden,
        lr,
        epochs,
        batch_size
    ));

    // Build the MLP: Linear(features, hidden) -> ReLU -> Linear(hidden, classes).
    let model = Sequential::new()
        .add(Linear::new(dataset.n_features, hidden, true))
        .add(ReLU::new())
        .add(Linear::new(hidden, n_classes, true));

    let trainer = TrainerBuilder::default()
        .model(model)
        .learning_rate(lr)
        .build();
    let mut trainer = match trainer {
        Ok(t) => t,
        Err(err) => {
            sink.println(&format!("error: {}", err));
            return ExitCode::ERROR;
        }
    };

    // Mini-batch the dataset.
    let n = dataset.features.len();
    let mut batches: Vec<Batch> = Vec::new();
    let mut start = 0;
    while start < n {
        let end = (start + batch_size).min(n);
        let mut batch_data = Vec::with_capacity((end - start) * dataset.n_features);
        for row in start..end {
            batch_data.extend_from_slice(&dataset.features[row]);
        }
        let inputs = Tensor::from_vec(batch_data, vec![end - start, dataset.n_features]);
        let targets = dataset.labels[start..end].to_vec();
        match Batch::new(inputs, targets) {
            Ok(b) => batches.push(b),
            Err(err) => {
                sink.println(&format!("error: {}", err));
                return ExitCode::ERROR;
            }
        }
        start = end;
    }

    let summary = match trainer.fit(&batches, epochs) {
        Ok(s) => s,
        Err(err) => {
            sink.println(&format!("error: {}", err));
            return ExitCode::ERROR;
        }
    };
    sink.println(&format!(
        "trained: loss={:.6} accuracy={:.3} steps={}",
        summary.loss, summary.accuracy, summary.steps
    ));

    // Serialize state + the architecture descriptor so `run` can reconstruct layers.
    let mut state: ModelState = trainer.state();
    state
        .metadata
        .insert("arch".to_string(), "linear,relu,linear".to_string());
    state
        .metadata
        .insert("n_features".to_string(), dataset.n_features.to_string());
    state
        .metadata
        .insert("classes".to_string(), n_classes.to_string());

    let bytes = state.to_brain_bytes();
    if let Err(err) = std::fs::write(std::path::Path::new(&out_path), &bytes) {
        sink.println(&format!("error: could not write '{}': {}", out_path, err));
        return ExitCode::IO_ERROR;
    }
    sink.println(&format!("wrote model checkpoint: {} ({} bytes)", out_path, bytes.len()));
    ExitCode::SUCCESS
}
