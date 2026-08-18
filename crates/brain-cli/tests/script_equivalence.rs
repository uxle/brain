//! # Script Host & REPL Equivalence Tests

use brain_cli::core::{ExitCode, OutputSink};
use brain_cli::script::run_script;

#[test]
fn test_script_execution_equivalence() {
    let script = r#"
# Brain Script Program
x = ones(2, 2)
y = ones(2, 2)
z = add(x, y)
"#;
    let sink = OutputSink::memory();
    let code = run_script(script, &sink);
    assert_eq!(code, ExitCode::SUCCESS);

    let output = sink.captured().unwrap_or_default();
    assert!(output.contains("z = [2, 2]"), "Expected output to contain z = [2, 2], got: {}", output);
}
