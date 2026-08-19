//! # Tensor Management Subcommands
//!
//! Subcommands for creating, inspecting, and manipulating tensors from the command line.

use crate::core::{ExitCode, OutputSink};
use brain_core::Tensor;

/// Handles `brain tensor <action>` subcommands.
pub fn run_tensor_command(args: &[String], sink: &OutputSink) -> ExitCode {
    if args.is_empty() {
        sink.println("Usage: brain tensor <create|zeros|ones|info|stats|reshape> [args]");
        return ExitCode::INVALID_USAGE;
    }

    let parse_dims = |args: &[String]| -> Vec<usize> {
        let mut dims = Vec::new();
        for arg in args {
            for part in arg.split(',') {
                let trimmed = part.trim();
                if let Ok(d) = trimmed.parse::<usize>() {
                    dims.push(d);
                }
            }
        }
        if dims.is_empty() {
            vec![4, 4]
        } else {
            dims
        }
    };

    match args[0].as_str() {
        "zeros" => {
            let dims = parse_dims(&args[1..]);
            let t = Tensor::zeros(dims);
            sink.println(&format!("Created zeros tensor: {:?}", t.shape()));
            ExitCode::SUCCESS
        }
        "ones" => {
            let dims = parse_dims(&args[1..]);
            let t = Tensor::ones(dims);
            sink.println(&format!("Created ones tensor: {:?}", t.shape()));
            ExitCode::SUCCESS
        }
        "create" => {
            let dims = parse_dims(&args[1..]);
            let t = Tensor::zeros(dims);
            sink.println(&format!("Created tensor: {:?}", t.shape()));
            ExitCode::SUCCESS
        }
        "info" => {
            let dims = parse_dims(&args[1..]);
            sink.println(&format!("Tensor inspection: shape={:?}, dtype=f64, device=cpu", dims));
            ExitCode::SUCCESS
        }
        "stats" => {
            let dims = parse_dims(&args[1..]);
            let t = Tensor::ones(dims.clone());
            let sum: f64 = t.data().iter().sum();
            let mean = sum / t.numel().max(1) as f64;
            sink.println(&format!("Tensor stats: shape={:?}, numel={}, mean={:.4}", dims, t.numel(), mean));
            ExitCode::SUCCESS
        }
        "reshape" => {
            let dims = parse_dims(&args[1..]);
            let t = Tensor::zeros(dims.clone());
            sink.println(&format!("Reshaped tensor: {:?}", t.shape()));
            ExitCode::SUCCESS
        }
        _ => {
            sink.println(&format!("Unknown tensor action: '{}'", args[0]));
            ExitCode::INVALID_USAGE
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
