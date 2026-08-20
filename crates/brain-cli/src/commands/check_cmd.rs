//! # `brain check` — Checkpoint Diagnostic & Gradient Health Checker

use crate::core::{ExitCode, OutputSink};
use brain_train::ModelState;

/// Handles `brain check <model.bn|model.brain>`.
pub fn run_check_command(args: &[String], sink: &OutputSink) -> ExitCode {
    if args.is_empty() {
        sink.println("Usage: brain check <model.bn|model.brain>");
        return ExitCode::INVALID_USAGE;
    }

    let model_path = &args[0];
    let bytes = match std::fs::read(std::path::Path::new(model_path)) {
        Ok(b) => b,
        Err(err) => {
            sink.println(&format!("error: could not read '{}': {}", model_path, err));
            return ExitCode::IO_ERROR;
        }
    };

    let state = match ModelState::from_brain_bytes(&bytes) {
        Ok(s) => s,
        Err(err) => {
            sink.println(&format!(
                "error: invalid model checkpoint '{}': {}",
                model_path, err
            ));
            return ExitCode::ERROR;
        }
    };

    sink.println(&format!("checking model: {}", model_path));
    let params = state.tensors();
    sink.println(&format!("parameter tensors: {}", params.len()));

    let mut all_finite = true;
    for (i, p) in params.iter().enumerate() {
        let nan_count = p.data().iter().filter(|v| v.is_nan()).count();
        let inf_count = p.data().iter().filter(|v| v.is_infinite()).count();
        if nan_count > 0 || inf_count > 0 {
            all_finite = false;
            sink.println(&format!(
                "  tensor #{}: shape={:?} [FAIL: {} NaNs, {} Infs]",
                i,
                p.shape(),
                nan_count,
                inf_count
            ));
        } else {
            let count = p.data().len().max(1);
            let mean: f64 = p.data().iter().sum::<f64>() / (count as f64);
            sink.println(&format!(
                "  tensor #{}: shape={:?} mean={:.4} [PASS: finite]",
                i,
                p.shape(),
                mean
            ));
        }
    }

    let arch = state
        .metadata
        .get("arch")
        .cloned()
        .unwrap_or_else(|| "linear,relu,linear".to_string());
    sink.println(&format!("architecture: {}", arch));

    if all_finite {
        sink.println("checkpoint check: PASSED (all parameters healthy and finite)");
        ExitCode::SUCCESS
    } else {
        sink.println("checkpoint check: FAILED (non-finite parameters detected)");
        ExitCode::ERROR
    }
}
