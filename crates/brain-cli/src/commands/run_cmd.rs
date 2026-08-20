//! `brain run` — load a checkpoint and run inference.

use crate::core::{ExitCode, OutputSink};
use crate::datafile::{load_task, Dataset, DatasetTask};

use brain_core::Tensor;
use brain_train::{Linear, ModelState, ReLU, Sequential, TrainableModule};

pub fn run_run_command(args: &[String], sink: &OutputSink) -> ExitCode {
    if args.is_empty() {
        sink.println("Usage: brain run <model.bn|model.brain> [--data DATA | --input \"a,b,c\"]");

        sink.println("Options:");

        sink.println("  --data PATH    Run inference over dataset");

        sink.println("  --input STR    Run one input sample");

        sink.println("  --top          Print only prediction");

        return ExitCode::INVALID_USAGE;
    }

    let model_path = args[0].clone();

    let mut data_path: Option<String> = None;
    let mut input_sample: Option<String> = None;
    let mut top_only = false;

    let mut i = 1usize;

    while i < args.len() {
        let arg = &args[i];

        if !arg.starts_with("--") {
            sink.println(&format!("error: unexpected argument '{}'", arg));

            return ExitCode::INVALID_USAGE;
        }

        let key = arg.trim_start_matches("--").split('=').next().unwrap_or("");

        let value = arg.split_once('=').map(|(_, v)| v);

        match key {
            "data" => {
                if let Some(v) = value {
                    data_path = Some(v.to_string());
                } else if let Some(next) = args.get(i + 1) {
                    if !next.starts_with("--") {
                        data_path = Some(next.to_string());
                        i += 1;
                    } else {
                        sink.println("error: '--data' requires a path argument");
                        return ExitCode::INVALID_USAGE;
                    }
                } else {
                    sink.println("error: '--data' requires a path argument");
                    return ExitCode::INVALID_USAGE;
                }
            }

            "input" => {
                if let Some(v) = value {
                    input_sample = Some(v.to_string());
                } else if let Some(next) = args.get(i + 1) {
                    if !next.starts_with("--") {
                        input_sample = Some(next.to_string());
                        i += 1;
                    } else {
                        sink.println("error: '--input' requires a sample argument");
                        return ExitCode::INVALID_USAGE;
                    }
                } else {
                    sink.println("error: '--input' requires a sample argument");
                    return ExitCode::INVALID_USAGE;
                }
            }

            "top" => {
                top_only = true;
            }

            _ => {
                sink.println(&format!("error: unknown option '{}'", arg));

                return ExitCode::INVALID_USAGE;
            }
        }

        i += 1;
    }

    let bytes = match std::fs::read(std::path::Path::new(&model_path)) {
        Ok(bytes) => bytes,

        Err(err) => {
            sink.println(&format!("error: could not read '{}': {}", model_path, err));

            return ExitCode::IO_ERROR;
        }
    };

    let state = match ModelState::from_brain_bytes(&bytes) {
        Ok(state) => state,

        Err(err) => {
            sink.println(&format!("error: invalid model '{}': {}", model_path, err));

            return ExitCode::ERROR;
        }
    };

    let task = state
        .metadata
        .get("task")
        .map(|s| s.as_str())
        .unwrap_or("classification");

    let arch = state
        .metadata
        .get("arch")
        .map(|s| s.as_str())
        .unwrap_or("linear");

    let params = state.tensors();

    let mut model = Sequential::new();

    let mut parameter_index = 0usize;
    let mut is_conv = false;

    for layer in arch.split(',') {
        match layer.trim() {
            "linear" => {
                if parameter_index >= params.len() {
                    sink.println("error: missing linear parameters");

                    return ExitCode::ERROR;
                }

                let weight = &params[parameter_index];

                if weight.ndim() != 2 {
                    sink.println("error: invalid linear weight tensor");

                    return ExitCode::ERROR;
                }

                let out_features = weight.shape()[0];

                let in_features = weight.shape()[1];

                let has_bias = parameter_index + 1 < params.len()
                    && params[parameter_index + 1].ndim() == 1
                    && params[parameter_index + 1].shape()[0] == out_features;

                model = model.add(Linear::new(in_features, out_features, has_bias));

                parameter_index += 1 + usize::from(has_bias);
            }

            "relu" => {
                model = model.add(ReLU::new());
            }

            "conv2d" => {
                is_conv = true;

                if parameter_index >= params.len() {
                    sink.println("error: missing conv2d parameters");

                    return ExitCode::ERROR;
                }

                let weight = &params[parameter_index];

                if weight.ndim() != 4 {
                    sink.println("error: invalid conv2d weight tensor");

                    return ExitCode::ERROR;
                }

                let out_channels = weight.shape()[0];

                let in_channels = weight.shape()[1];

                let kernel = weight.shape()[2];

                let has_bias = parameter_index + 1 < params.len()
                    && params[parameter_index + 1].ndim() == 1
                    && params[parameter_index + 1].shape()[0] == out_channels;

                model = model.add(brain_train::Conv2d::new(
                    in_channels,
                    out_channels,
                    kernel,
                    has_bias,
                ));

                parameter_index += 1 + usize::from(has_bias);
            }

            "maxpool2d" => {
                model = model.add(brain_train::MaxPool2d::new(2, 2));
            }

            "avgpool2d" => {
                model = model.add(brain_train::AvgPool2d::new(2, 2));
            }

            "flatten" => {
                model = model.add(brain_train::Flatten::new());
            }

            unknown => {
                sink.println(&format!("error: unknown layer '{}'", unknown));

                return ExitCode::ERROR;
            }
        }
    }

    if let Err(err) = model.load_parameters(&params) {
        sink.println(&format!("error: loading parameters: {}", err));

        return ExitCode::ERROR;
    }

    sink.println(&format!(
        "loaded model: {} with {} param tensors across {} layers",
        model_path,
        params.len(),
        model.len()
    ));

    /*
     * Build input.
     */
    let (raw_inputs, labels, targets) = if let Some(sample) = input_sample {
        let row = match Dataset::parse_sample(&sample) {
            Ok(row) => row,

            Err(err) => {
                sink.println(&format!("error: --input: {}", err));

                return ExitCode::INVALID_USAGE;
            }
        };

        (
            Tensor::from_vec(row.clone(), vec![1, row.len()]),
            None,
            None,
        )
    } else if let Some(path) = data_path {
        if task == "regression" {
            let dataset = match load_task(&path, DatasetTask::Regression) {
                Ok(d) => d,

                Err(err) => {
                    sink.println(&format!("error: {}", err));

                    return ExitCode::ERROR;
                }
            };

            (dataset.feature_matrix(), None, Some(dataset.targets))
        } else {
            let dataset = match load_task(&path, DatasetTask::Classification) {
                Ok(d) => d,

                Err(err) => {
                    sink.println(&format!("error: {}", err));

                    return ExitCode::ERROR;
                }
            };

            (dataset.feature_matrix(), Some(dataset.labels), None)
        }
    } else {
        sink.println("error: provide --data <file> or --input \"a,b,c\"");

        return ExitCode::INVALID_USAGE;
    };

    /*
     * Convert image-like input for convnet.
     */
    let inputs = if is_conv && raw_inputs.ndim() == 2 {
        let samples = raw_inputs.shape()[0];

        let features = raw_inputs.shape()[1];

        let side = (features as f64).sqrt().round() as usize;

        let side = if side * side == features && side >= 3 {
            side
        } else {
            6
        };

        let mut data = raw_inputs.to_vec();

        if data.len() == samples * side * side {
            Tensor::from_vec(data, vec![samples, 1, side, side])
        } else {
            data.resize(samples * 36, 0.0);

            Tensor::from_vec(data, vec![samples, 1, 6, 6])
        }
    } else {
        raw_inputs
    };

    /*
     * Forward pass.
     */
    let output = match model.forward(&inputs) {
        Ok(output) => output,

        Err(err) => {
            sink.println(&format!("error: forward pass: {}", err));

            return ExitCode::ERROR;
        }
    };

    if output.ndim() != 2 {
        sink.println(&format!(
            "error: expected 2D model output, got {:?}",
            output.shape()
        ));

        return ExitCode::ERROR;
    }

    let samples = output.shape()[0];

    let outputs = output.shape()[1];

    /*
     * REGRESSION
     */
    if task == "regression" {
        if outputs != 1 {
            sink.println(&format!(
                "error: regression model must have 1 output, got {}",
                outputs
            ));

            return ExitCode::ERROR;
        }

        if top_only {
            for row in 0..samples {
                sink.println(&format!("prediction: {:.6}", output.get(row)));
            }

            return ExitCode::SUCCESS;
        }

        sink.println(&format!("predictions [{} x 1]:", samples));

        for row in 0..samples {
            let prediction = output.get(row);

            if let Some(ts) = &targets {
                sink.println(&format!(
                    "  sample {}: {:.6} (actual: {:.6})",
                    row, prediction, ts[row]
                ));
            } else {
                sink.println(&format!("  sample {}: {:.6}", row, prediction));
            }
        }

        if let Some(ts) = &targets {
            let mut mse = 0.0;

            for row in 0..samples {
                let error = output.get(row) - ts[row];

                mse += error * error;
            }

            mse /= samples.max(1) as f64;

            sink.println(&format!("mse: {:.6}", mse));
        }

        return ExitCode::SUCCESS;
    }

    /*
     * CLASSIFICATION
     */
    let mut predictions = Vec::<usize>::with_capacity(samples);

    for row in 0..samples {
        let offset = row * outputs;

        let mut best = 0usize;

        let mut best_value = output.get(offset);

        for col in 1..outputs {
            let value = output.get(offset + col);

            if value > best_value {
                best_value = value;
                best = col;
            }
        }

        predictions.push(best);
    }

    if top_only {
        for prediction in &predictions {
            sink.println(&format!("predicted class: {}", prediction));
        }

        return ExitCode::SUCCESS;
    }

    sink.println(&format!("logits [{} x {}]:", samples, outputs));

    for row in 0..samples {
        let offset = row * outputs;

        let values = (0..outputs)
            .map(|col| format!("{:+.4}", output.get(offset + col)))
            .collect::<Vec<_>>();

        let mut line = format!(
            "  sample {}: [{}] -> class {}",
            row,
            values.join(", "),
            predictions[row]
        );

        if let Some(lbls) = &labels {
            line.push_str(&format!(" (actual: {})", lbls[row]));
        }

        sink.println(&line);
    }

    if let Some(lbls) = &labels {
        let correct = predictions
            .iter()
            .zip(lbls.iter())
            .filter(|(prediction, label)| **prediction == **label)
            .count();

        let accuracy = correct as f64 / samples.max(1) as f64;

        sink.println(&format!(
            "accuracy: {:.3} ({}/{})",
            accuracy, correct, samples
        ));
    }

    ExitCode::SUCCESS
}
