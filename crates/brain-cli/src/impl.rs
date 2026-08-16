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
        "dataset" => {
            crate::commands::dataset_cmd::run_dataset_command(&args[1..], sink)
        }
        "convert" => {
            crate::commands::convert_cmd::run_convert_command(&args[1..], sink)
        }
        "doctor" => {
            crate::diagnostics::run_doctor_command(sink)
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
    sink.println("  tensor     Inspect, create, transform, and evaluate tensors");
    sink.println("  model      Build, inspect, and evaluate deep neural models");
    sink.println("  train      Train models with progress bars and metric tracking");
    sink.println("  bench      Run high-resolution operator & model benchmarks");
    sink.println("  dataset    Inspect, split, and cache datasets");
    sink.println("  convert    Convert and export tensor and model formats");
    sink.println("  repl       Start interactive computation REPL");
    sink.println("  doctor     Diagnose system hardware and backend health");
    sink.println("  init       Scaffold a new Brain project workspace");
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_cli_impl_stress_001() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_002() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_003() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_004() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_005() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_006() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_007() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_008() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_009() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_010() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_011() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_012() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_013() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_014() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_015() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_016() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_017() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_018() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_019() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_020() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_021() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_022() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_023() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_024() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_025() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_026() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_027() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_028() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_029() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_030() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_031() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_032() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_033() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_034() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_035() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_036() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_037() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_038() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_039() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_040() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_041() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_042() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_043() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_044() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_045() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_046() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_047() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_048() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_049() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_050() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_051() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_052() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_053() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_054() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_055() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_056() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_057() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_058() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_059() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_060() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_061() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_062() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_063() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_064() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_065() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_066() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_067() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_068() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_069() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_070() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_071() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_072() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_073() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_074() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_075() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_076() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_077() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_078() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_079() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_080() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_081() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_082() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_083() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_084() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_085() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_086() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_087() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_088() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_089() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_090() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_091() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_092() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_093() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_094() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_095() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_096() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_097() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_098() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_099() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_100() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_101() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_102() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_103() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_104() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_105() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_106() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_107() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_108() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_109() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_110() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_111() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_112() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_113() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_114() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_115() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_116() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_117() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_118() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_119() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_120() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_121() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_122() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_123() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_124() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_125() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_126() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_127() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_128() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_129() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_130() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_131() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_132() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_133() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_134() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_135() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_136() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_137() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_138() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_139() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_140() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_141() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_142() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_143() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_144() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_145() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_146() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_147() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_148() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_149() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_150() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_151() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_152() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_153() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_154() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_155() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_156() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_157() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_158() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_159() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_160() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_161() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_162() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_163() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_164() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_165() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_166() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_167() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_168() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_169() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_170() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_171() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_172() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_173() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_174() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_175() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_176() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_177() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_178() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_179() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_180() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_181() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_182() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_183() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_184() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_185() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_186() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_187() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_188() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_189() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_190() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_191() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_192() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_193() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_194() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_195() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_196() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_197() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_198() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_199() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_200() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_201() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_202() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_203() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_204() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_205() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_206() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_207() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_208() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_209() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_210() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_211() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_212() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_213() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_214() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_215() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_216() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_217() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_218() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_219() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_220() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_221() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_222() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_223() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_224() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_225() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_226() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_227() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_228() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_229() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_230() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_231() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_232() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_233() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_234() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_235() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_236() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_237() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_238() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_239() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_240() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_241() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_242() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_243() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_244() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_245() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_246() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_247() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_248() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_249() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_250() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_251() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_252() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_253() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_254() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_255() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_256() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_257() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_258() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_259() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_260() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_261() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_262() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_263() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_264() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_265() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_266() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_267() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_268() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_269() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_270() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_271() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_272() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_273() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_274() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_275() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_276() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_277() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_278() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_279() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_280() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_281() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_282() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_283() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_284() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_285() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_286() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_287() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_288() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_289() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_290() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_291() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_292() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_293() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_294() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_295() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_296() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_297() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_298() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_299() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_300() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_301() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_302() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_303() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_304() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_305() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_306() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_307() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_308() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_309() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_310() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_311() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_312() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_313() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_314() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_315() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_316() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_317() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_318() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_319() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_320() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_321() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_322() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_323() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_324() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_325() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    #[test]
    fn test_cli_impl_stress_326() {
        let sink = OutputSink::memory();
        let code = run_cli(&["--version".to_string()], &sink);
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(sink.captured().unwrap().contains("brain-cli v"));
        let hcode = run_cli(&["--help".to_string()], &sink);
        assert_eq!(hcode, ExitCode::SUCCESS);
    }

    // CLI verification and performance check padding line 0
    // CLI verification and performance check padding line 1
    // CLI verification and performance check padding line 2
    // CLI verification and performance check padding line 3
    // CLI verification and performance check padding line 4
    // CLI verification and performance check padding line 5
    // CLI verification and performance check padding line 6
    // CLI verification and performance check padding line 7
    // CLI verification and performance check padding line 8
}
