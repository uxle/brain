//! # Main CLI Dispatcher & Router
//!
//! Entry point for argument routing, help and version rendering, and top-level execution handling.

use crate::core::{ExitCode, OutputSink};

/// Executes the CLI command loop given command-line arguments.
pub fn run_cli(args: &[String], sink: &OutputSink) -> ExitCode {
    if args.is_empty() {
        print_help(sink);
        return ExitCode::SUCCESS;
    }

    let first = args[0].as_str();
    match first {
        "--version" | "-V" | "version" => {
            print_version(sink);
            ExitCode::SUCCESS
        }
        "--help" | "-h" | "help" => {
            print_help(sink);
            ExitCode::SUCCESS
        }
        "agent" => {
            crate::commands::agent_cmd::run_agent_command(&args[1..], sink)
        }
        "tensor" => {
            crate::commands::tensor_cmd::run_tensor_command(&args[1..], sink)
        }
        "bench" => {
            crate::commands::bench_cmd::run_bench_command(&args[1..], sink)
        }
        "model" => {
            crate::commands::model_cmd::run_model_command(&args[1..], sink)
        }
        "train" => {
            crate::commands::train_cmd::run_train_command(&args[1..], sink)
        }
        "make" => {
            crate::commands::make_cmd::run_make_command(&args[1..], sink)
        }
        "run" => {
            crate::commands::run_cmd::run_run_command(&args[1..], sink)
        }
        "space" => {
            crate::commands::space_cmd::run_space_command(&args[1..], sink)
        }
        "chatbot" | "chat" => {
            let mut space_args = vec!["chatbot".to_string()];
            space_args.extend_from_slice(&args[1..]);
            crate::commands::space_cmd::run_space_command(&space_args, sink)
        }
        "new" => {
            let mut space_args = vec!["new".to_string()];
            space_args.extend_from_slice(&args[1..]);
            crate::commands::space_cmd::run_space_command(&space_args, sink)
        }
        "check" => {
            crate::commands::check_cmd::run_check_command(&args[1..], sink)
        }
        "script" => {
            if args.len() < 2 {
                sink.println("Usage: brain script <file.brain>");
                ExitCode::INVALID_USAGE
            } else {
                let file_path = &args[1];
                match std::fs::read_to_string(file_path) {
                    Ok(content) => crate::script::run_script(&content, sink),
                    Err(err) => {
                        sink.println(&format!("error: could not read '{}': {}", file_path, err));
                        ExitCode::IO_ERROR
                    }
                }
            }
        }
        "dataset" => {
            crate::commands::dataset_cmd::run_dataset_command(&args[1..], sink)
        }
        "convert" => {
            crate::commands::convert_cmd::run_convert_command(&args[1..], sink)
        }
        "doctor" => {
            crate::diagnostics::run_doctor_command(sink)
        }
        "repl" => {
            crate::repl::run_repl_command(sink)
        }
        "init" => {
            let name = args.get(1).map(|s| s.as_str()).unwrap_or("brain_project");
            crate::init::scaffold_project(name, sink)
        }
        unknown => {
            sink.println(&format!("Unknown command: '{}'. Run 'brain --help' for usage.", unknown));
            ExitCode::INVALID_USAGE
        }
    }
}

/// Prints formatted version information.
pub fn print_version(sink: &OutputSink) {
    sink.println(&format!("brain-cli v{}", crate::VERSION));
}

/// Prints top-level CLI help and subcommand descriptions.
pub fn print_help(sink: &OutputSink) {
    sink.println("Brain Deep Learning Framework CLI");
    sink.println("Usage: brain <command> [options]");
    sink.println("");
    sink.println("Commands:");
    sink.println("  new        Create a newborn 3D cubic neural mind in a .bn file");
    sink.println("  chat       Start an interactive conversation with a growing BrainMind");
    sink.println("  make       Build, train, and checkpoint a model from a dataset");
    sink.println("  run        Load a model checkpoint and run inference");
    sink.println("  agent      Run autonomous perceive-think-act-learn agent loop");
    sink.println("  train      Train models with progress bars and metric tracking");
    sink.println("  model      Build, inspect, and evaluate deep neural models");
    sink.println("  check      Verify model and computational graph integrity");
    sink.println("  tensor     Inspect, create, transform, and evaluate tensors");
    sink.println("  bench      Run high-resolution operator & model benchmarks");
    sink.println("  dataset    Inspect, split, and cache datasets");
    sink.println("  convert    Convert and export tensor and model formats");
    sink.println("  repl       Start interactive computation REPL");
    sink.println("  script     Execute declarative .brain automation scripts");
    sink.println("  doctor     Diagnose system hardware and backend health");
    sink.println("  init       Scaffold a new Brain project workspace");
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
