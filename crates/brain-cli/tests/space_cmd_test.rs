//! # 3D Cubic Brain Space CLI Integration Test

use brain_cli::core::OutputSink;
use brain_cli::run_cli;
use brain_core::BrainModelFile;

#[test]
fn test_brain_space_cli_generation_and_load() {
    let test_file = "target/test_cube_brain.bn";
    let _ = std::fs::remove_file(test_file);

    let sink = OutputSink::memory();
    let exit_code = run_cli(
        &[
            "space".to_string(),
            test_file.to_string(),
            "--cube".to_string(),
            "5".to_string(),
        ],
        &sink,
    );

    let output = sink.captured().unwrap_or_default();
    println!("Sink output: {}", output);
    assert!(
        exit_code.is_success(),
        "Failed with exit code {:?}, output: {}",
        exit_code,
        output
    );
    assert!(output.contains("5 x 5 x 5"));
    assert!(output.contains("125"));

    // Verify .bn file on disk
    let loaded = BrainModelFile::load_file(test_file).expect("Load generated .bn file");
    assert_eq!(loaded.name, "growing_brain");
    assert_eq!(loaded.metadata.get("cube_dim").unwrap(), "5");
    assert_eq!(loaded.metadata.get("total_neurons").unwrap(), "125");

    let _ = std::fs::remove_file(test_file);
}

#[test]
fn test_brain_new_cli_generation() {
    let test_file = "target/test_direct_new_brain.bn";
    let _ = std::fs::remove_file(test_file);

    let sink = OutputSink::memory();
    let exit_code = run_cli(
        &[
            "new".to_string(),
            test_file.to_string(),
            "--cube".to_string(),
            "4".to_string(),
        ],
        &sink,
    );

    let output = sink.captured().unwrap_or_default();
    assert!(exit_code.is_success());
    assert!(output.contains("4 x 4 x 4"));
    assert!(output.contains("64"));

    let _ = std::fs::remove_file(test_file);
}
