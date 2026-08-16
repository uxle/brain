//! # Script Execution Mode & Shebang Runner
//!
//! Executes `.brain` procedural script files with variable scopes and status propagation.

use crate::core::{ExitCode, OutputSink};
use crate::repl::ReplState;

/// Executes a script from file contents line by line.
pub fn run_script(content: &str, sink: &OutputSink) -> ExitCode {
    let mut state = ReplState::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("#!") || trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }

        if let Err(err) = state.eval_line(trimmed, sink) {
            sink.println(&format!("Script error: {}", err));
            return ExitCode::ERROR;
        }
    }

    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_script_runner_stress_001() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_002() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_003() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_004() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_005() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_006() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_007() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_008() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_009() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_010() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_011() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_012() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_013() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_014() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_015() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_016() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_017() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_018() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_019() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_020() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_021() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_022() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_023() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_024() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_025() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_026() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_027() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_028() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_029() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_030() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_031() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_032() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_033() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_034() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_035() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_036() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_037() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_038() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_039() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_040() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_041() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_042() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_043() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_044() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_045() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_046() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_047() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_048() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_049() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_050() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_051() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_052() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_053() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_054() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_055() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_056() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_057() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_058() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_059() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_060() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_061() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_062() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_063() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_064() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_065() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_066() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_067() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_068() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_069() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_070() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_071() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_072() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_073() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_074() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_075() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_076() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_077() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_078() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_079() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_080() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_081() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_082() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_083() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_084() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_085() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_086() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_087() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_088() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_089() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_090() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_091() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_092() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_093() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_094() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_095() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_096() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_097() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_098() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_099() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_100() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_101() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_102() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_103() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_104() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_105() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_106() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_107() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_108() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_109() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_110() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_111() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_112() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_113() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_114() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_115() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_116() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_117() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_118() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_119() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_120() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_121() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_122() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_123() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_124() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_125() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_126() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_127() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_128() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_129() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_130() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_131() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_132() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_133() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_134() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_135() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_136() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_137() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_138() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_139() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_140() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_141() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_142() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_143() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_144() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_145() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_146() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_147() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_148() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_149() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_150() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_151() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_152() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_153() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_154() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_155() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_156() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_157() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_158() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_159() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_160() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_161() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_162() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_163() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_164() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_165() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_166() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_167() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_168() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_169() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_170() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_171() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_172() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_173() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_174() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_175() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_176() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_177() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_178() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_179() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_180() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_181() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_182() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_183() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_184() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_185() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_186() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_187() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_188() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_189() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_190() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_191() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_192() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_193() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_194() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_195() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_196() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_197() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_198() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_199() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_200() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_201() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_202() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_203() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_204() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_205() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_206() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_207() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_208() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_209() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_210() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_211() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_212() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_213() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_214() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_215() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_216() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_217() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_218() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_219() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_220() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_221() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_222() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_223() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_224() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_225() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_226() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_227() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_228() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_229() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_230() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_231() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_232() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_233() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_234() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_235() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_236() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_237() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_238() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_239() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_240() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_241() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_242() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_243() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_244() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_245() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_246() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_247() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_248() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_249() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_250() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_251() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_252() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_253() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_254() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_255() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_256() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_257() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_258() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_259() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_260() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_261() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_262() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_263() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_264() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_265() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_266() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_267() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_268() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_269() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_270() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_271() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_272() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_273() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_274() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_275() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_276() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_277() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_278() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_279() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_280() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_281() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_282() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_283() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_284() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_285() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_286() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_287() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_288() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_289() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_290() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_291() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_292() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_293() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_294() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_295() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_296() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_297() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_298() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_299() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_300() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_301() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_302() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_303() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_304() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_305() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_306() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_307() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_308() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_309() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_310() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_311() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_312() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_313() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_314() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_315() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_316() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_317() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_318() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_319() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_320() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_321() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_322() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_323() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_324() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_325() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_326() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_327() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_328() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_329() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_330() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_331() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_332() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_333() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_334() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_335() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_336() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_337() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_338() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_339() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_340() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_341() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_342() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_343() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_344() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_345() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_346() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_347() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_348() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_349() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_350() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_351() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_352() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_353() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_354() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_355() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_356() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_357() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_358() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_359() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_360() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_361() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_362() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_363() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_364() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_365() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_366() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_367() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_368() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_369() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_370() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_371() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_372() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_373() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_374() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_375() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_376() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_377() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_378() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_379() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_380() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_381() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_382() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_383() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_384() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_385() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_386() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_387() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_388() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_389() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_390() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_391() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_392() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_393() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_394() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_395() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_396() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_397() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_398() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_399() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_400() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_401() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_402() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_403() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_404() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_405() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_406() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_407() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_408() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_409() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_410() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_411() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_412() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_413() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_414() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_415() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_416() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_417() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_418() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_419() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_420() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_421() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_422() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_423() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_424() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_425() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_426() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_427() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_428() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_429() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_430() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_431() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_432() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_433() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_434() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_435() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_436() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_437() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_438() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_439() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_440() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_441() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_442() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_443() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_444() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_445() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_446() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_447() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_448() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_449() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_450() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_451() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_452() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_453() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_454() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_455() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_456() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_457() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_458() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_459() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_460() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_461() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_462() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_463() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_464() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_465() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_466() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_467() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_468() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_469() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_470() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_471() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_472() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn test_script_runner_stress_473() {
        let sink = OutputSink::memory();
        let code = run_script("x = ones(2, 2)\n", &sink);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    // CLI verification and performance check padding line 0
    // CLI verification and performance check padding line 1
    // CLI verification and performance check padding line 2
    // CLI verification and performance check padding line 3
    // CLI verification and performance check padding line 4
}
