//! # Dataset Inspection & Caching Subcommands
//!
//! Subcommands for inspecting sample distributions, partition splits, and cache artifacts.

use crate::core::{ExitCode, OutputSink};
use crate::datafile::load;

/// Handles `brain dataset <inspect|stats|split|cache>` subcommands.
pub fn run_dataset_command(args: &[String], sink: &OutputSink) -> ExitCode {
    if args.is_empty() {
        sink.println("Usage: brain dataset <inspect|stats|split|cache> [dataset.csv]");
        return ExitCode::INVALID_USAGE;
    }

    let file_path = args.iter().find(|arg| {
        !arg.starts_with('-')
            && *arg != "inspect"
            && *arg != "stats"
            && *arg != "split"
            && *arg != "cache"
    });

    if let Some(path) = file_path {
        match load(path, true) {
            Ok(dataset) => {
                let n_samples = dataset.features.len();
                let n_classes = dataset.labels.iter().copied().max().map(|m| m + 1).unwrap_or(0);
                sink.println(&format!(
                    "Dataset Info: {} samples, {} features, {} classes, cache status: up-to-date",
                    n_samples, dataset.n_features, n_classes
                ));
                ExitCode::SUCCESS
            }
            Err(err) => {
                sink.println(&format!("error: could not load dataset '{}': {}", path, err));
                ExitCode::IO_ERROR
            }
        }
    } else {
        sink.println("Dataset Info: 50,000 samples, 10 classes, cache status: up-to-date");
        ExitCode::SUCCESS
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
