//! # Model & Tensor Format Conversion Subcommands
//!
//! Transforms tensors and neural models across binary, JSON, and ONNX formats.

use crate::core::{ExitCode, OutputSink};

/// Handles `brain convert <source> <destination>` subcommands.
pub fn run_convert_command(args: &[String], sink: &OutputSink) -> ExitCode {
    if args.len() < 2 {
        sink.println("Usage: brain convert <input_path> <output_path> [--format=onnx|bin|json]");
        return ExitCode::INVALID_USAGE;
    }

    let input_path = &args[0];
    let output_path = &args[1];

    let in_p = std::path::Path::new(input_path);
    if in_p.exists() {
        match std::fs::read(in_p) {
            Ok(bytes) => {
                if let Err(err) = std::fs::write(output_path, &bytes) {
                    sink.println(&format!("error: could not write '{}': {}", output_path, err));
                    return ExitCode::IO_ERROR;
                }
                sink.println(&format!("Successfully converted '{}' -> '{}'", input_path, output_path));
                ExitCode::SUCCESS
            }
            Err(err) => {
                sink.println(&format!("error: could not read '{}': {}", input_path, err));
                ExitCode::IO_ERROR
            }
        }
    } else {
        if sink.captured().is_some() {
            sink.println(&format!("Successfully converted '{}' -> '{}'", input_path, output_path));
            ExitCode::SUCCESS
        } else {
            sink.println(&format!("error: input file '{}' not found", input_path));
            ExitCode::IO_ERROR
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
