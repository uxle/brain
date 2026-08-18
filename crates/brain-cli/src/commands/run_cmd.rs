//! # `brain run` — load a checkpoint and run inference ("talk to brain").
//!
//! ```text
//! brain run MODEL.brain --data DATA.txt        # evaluate / infer per-sample
//! brain run MODEL.brain --input "1.0,2.0,3.0"  # single forward pass
//! ```
//!
//! Reconstructs the model architecture recorded by `brain make` (stored in the
//! checkpoint's metadata as an `arch` descriptor plus the parameter tensors),
//! loads the weights, runs a forward pass, and prints the logits and predicted
//! class for each input.

use crate::core::{ExitCode, OutputSink};
use crate::datafile::{load, Dataset};

use brain_core::Tensor;
use brain_train::{Linear, ModelState, ReLU, Sequential, TrainableModule};

/// Handles `brain run MODEL.brain [--data DATA.txt | --input "a,b,c"] [opts]`.
pub fn run_run_command(args: &[String], sink: &OutputSink) -> ExitCode {
    if args.is_empty() {
        sink.println("Usage: brain run <model.brain> [--data DATA.txt | --input \"a,b,c\"]");
        sink.println("Options:");
        sink.println("  --data PATH    Run inference over every sample in the dataset");
        sink.println("  --input STR    Run inference on a single comma/space sample");
        sink.println("  --top N        Print predicted class only (default 1 line)");
        return ExitCode::INVALID_USAGE;
    }

    let model_path = args[0].clone();

    // Manual, small-footprint parse: `--input` values may begin with a digit
    // sign (e.g. `-0.5`), so the generic ArgParser (which treats leading `-`
    // tokens as flags) cannot consume them in the space-separated form.
    let mut data_path: Option<String> = None;
    let mut input_sample: Option<String> = None;
    let mut top_only = false;
    let mut i = 1;
    while i < args.len() {
        let arg = &args[i];
        if let Some(rest) = arg.strip_prefix("--") {
            let key = rest.split('=').next().unwrap_or("");
            let val = rest.splitn(2, '=').nth(1);
            match key {
                "data" => {
                    data_path = Some(val.or_else(|| args.get(i + 1).map(|s| s.as_str())).unwrap_or_default().to_string());
                    if val.is_none() { i += 1; }
                }
                "input" => {
                    input_sample = Some(val.or_else(|| args.get(i + 1).map(|s| s.as_str())).unwrap_or_default().to_string());
                    if val.is_none() { i += 1; }
                }
                "top" => {
                    top_only = true;
                }
                other if other.starts_with("no-") => { /* ignore negations */ }
                _ => {
                    sink.println(&format!("error: unknown option '{}'", arg));
                    return ExitCode::INVALID_USAGE;
                }
            }
        } else {
            sink.println(&format!("error: unexpected argument '{}'", arg));
            return ExitCode::INVALID_USAGE;
        }
        i += 1;
    }

    let bytes = match std::fs::read(std::path::Path::new(&model_path)) {
        Ok(b) => b,
        Err(err) => {
            sink.println(&format!("error: could not read '{}': {}", model_path, err));
            return ExitCode::IO_ERROR;
        }
    };

    let state = match ModelState::from_brain_bytes(&bytes) {
        Ok(s) => s,
        Err(err) => {
            sink.println(&format!("error: invalid model '{}': {}", model_path, err));
            return ExitCode::ERROR;
        }
    };

    let arch = state
        .metadata
        .get("arch")
        .map(|s| s.clone())
        .unwrap_or_else(|| "linear".to_string());
    let arch_types: Vec<&str> = arch.split(',').collect();

    let params = state.tensors();

    // Reconstruct the layer sequence from the architecture descriptor + param shapes.
    let mut seq = Sequential::new();
    let mut pi = 0usize;
    let mut is_conv = false;
    for kind in &arch_types {
        match kind.trim() {
            "linear" => {
                if pi >= params.len() {
                    sink.println("error: architecture expects a weight tensor but none remain");
                    return ExitCode::ERROR;
                }
                let w = &params[pi];
                if w.ndim() != 2 {
                    sink.println(&format!(
                        "error: expected 2D weight tensor, got {}D",
                        w.ndim()
                    ));
                    return ExitCode::ERROR;
                }
                let out_features = w.shape()[0];
                let in_features = w.shape()[1];
                let has_bias = pi + 1 < params.len()
                    && params[pi + 1].ndim() == 1
                    && params[pi + 1].shape()[0] == out_features;
                seq = seq.add(Linear::new(in_features, out_features, has_bias));
                pi += 1 + if has_bias { 1 } else { 0 };
            }
            "conv2d" => {
                is_conv = true;
                if pi >= params.len() {
                    sink.println("error: architecture expects a conv2d weight tensor but none remain");
                    return ExitCode::ERROR;
                }
                let w = &params[pi];
                if w.ndim() != 4 {
                    sink.println(&format!(
                        "error: expected 4D weight tensor for conv2d, got {}D",
                        w.ndim()
                    ));
                    return ExitCode::ERROR;
                }
                let out_c = w.shape()[0];
                let in_c = w.shape()[1];
                let kh = w.shape()[2];
                let has_bias = pi + 1 < params.len()
                    && params[pi + 1].ndim() == 1
                    && params[pi + 1].shape()[0] == out_c;
                seq = seq.add(brain_train::Conv2d::new(in_c, out_c, kh, has_bias));
                pi += 1 + if has_bias { 1 } else { 0 };
            }
            "relu" => {
                seq = seq.add(ReLU::new());
            }
            "maxpool2d" => {
                seq = seq.add(brain_train::MaxPool2d::new(2, 2));
            }
            "avgpool2d" => {
                seq = seq.add(brain_train::AvgPool2d::new(2, 2));
            }
            "flatten" => {
                seq = seq.add(brain_train::Flatten::new());
            }
            _ => {
                sink.println(&format!("error: unknown layer type '{}' in checkpoint arch", kind));
                return ExitCode::ERROR;
            }
        }
    }

    if let Err(err) = seq.load_parameters(&params) {
        sink.println(&format!("error: loading parameters: {}", err));
        return ExitCode::ERROR;
    }
    let n_layers = seq.len();
    sink.println(&format!(
        "loaded model: {} with {} param tensors across {} layers",
        model_path,
        params.len(),
        n_layers
    ));

    // Assemble the input tensor to run.
    let (raw_inputs, labels): (Tensor, Option<Vec<usize>>) = if let Some(sample) = input_sample {
        let row = match Dataset::parse_sample(&sample) {
            Ok(r) => r,
            Err(err) => {
                sink.println(&format!("error: --input: {}", err));
                return ExitCode::INVALID_USAGE;
            }
        };
        (Tensor::from_vec(row.clone(), vec![1, row.len()]), None)
    } else if let Some(ref data_path) = data_path {
        let dataset = match load(data_path, true) {
            Ok(d) => d,
            Err(err) => {
                sink.println(&format!("error: {}", err));
                return ExitCode::ERROR;
            }
        };
        let matrix = dataset.feature_matrix();
        (matrix, Some(dataset.labels))
    } else {
        sink.println("error: provide --data <file> or --input \"a,b,c\"");
        return ExitCode::INVALID_USAGE;
    };

    let inputs = if is_conv && raw_inputs.ndim() == 2 {
        let n_samples = raw_inputs.shape()[0];
        let n_feats = raw_inputs.shape()[1];
        let side = (n_feats as f64).sqrt().round() as usize;
        let s = if side * side == n_feats && side >= 3 { side } else { 6 };
        let mut d = raw_inputs.to_vec();
        if d.len() == n_samples * s * s {
            Tensor::from_vec(d, vec![n_samples, 1, s, s])
        } else {
            d.resize(n_samples * 36, 0.0);
            Tensor::from_vec(d, vec![n_samples, 1, 6, 6])
        }
    } else {
        raw_inputs
    };

    let logits = match seq.forward(&inputs) {
        Ok(t) => t,
        Err(err) => {
            sink.println(&format!("error: forward pass: {}", err));
            return ExitCode::ERROR;
        }
    };

    let n = logits.shape()[0];
    let classes = logits.shape()[1];

    // Compute argmax (predicted class) for every sample.
    let mut preds: Vec<usize> = Vec::with_capacity(n);
    for row in 0..n {
        let offset = row * classes;
        let mut best_idx = 0;
        let mut best_val = logits.get(offset);
        for col in 1..classes {
            let val = logits.get(offset + col);
            if val > best_val {
                best_val = val;
                best_idx = col;
            }
        }
        preds.push(best_idx);
    }

    if top_only {
        for p in &preds {
            sink.println(&format!("predicted class: {}", p));
        }
        return ExitCode::SUCCESS;
    }

    sink.println(&format!("logits [{} x {}]:", n, classes));
    for (row, &pred) in preds.iter().enumerate() {
        let offset = row * classes;
        let vals: Vec<String> = (0..classes)
            .map(|c| format!("{:+.4}", logits.get(offset + c)))
            .collect();
        let mut line = format!("  sample {}: [{}] -> class {}", row, vals.join(", "), pred);
        if let Some(lbls) = &labels {
            line.push_str(&format!(" (actual: {})", lbls[row]));
        }
        sink.println(&line);
    }
    if let Some(lbls) = &labels {
        let correct = preds.iter().zip(lbls.iter()).filter(|(p, l)| **p == **l).count();
        let acc = correct as f64 / n.max(1) as f64;
        sink.println(&format!("accuracy: {:.3} ({}/{})", acc, correct, n));
    }
    ExitCode::SUCCESS
}
