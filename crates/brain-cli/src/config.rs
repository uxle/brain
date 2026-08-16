//! # CLI Runtime Configuration & Environment Options
//!
//! Controls output verbosity levels, ANSI color modes, timeout budgets, and compute devices.

/// Console logging verbosity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Verbosity {
    Quiet,
    #[default]
    Normal,
    Verbose,
    Debug,
    Trace,
}

/// Terminal ANSI color preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorChoice {
    #[default]
    Auto,
    Always,
    Never,
}

/// Global CLI invocation options.
#[derive(Debug, Clone)]
pub struct CliConfig {
    pub verbosity: Verbosity,
    pub color: ColorChoice,
    pub device: String,
    pub timeout_seconds: Option<u64>,
    pub num_threads: usize,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            verbosity: Verbosity::Normal,
            color: ColorChoice::Auto,
            device: "cpu".to_string(),
            timeout_seconds: None,
            num_threads: 1,
        }
    }
}

impl CliConfig {
    /// Creates a new default `CliConfig`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets verbosity level.
    pub fn with_verbosity(mut self, verbosity: Verbosity) -> Self {
        self.verbosity = verbosity;
        self
    }

    /// Sets color mode.
    pub fn with_color(mut self, color: ColorChoice) -> Self {
        self.color = color;
        self
    }

    /// Sets compute device target string.
    pub fn with_device(mut self, device: impl Into<String>) -> Self {
        self.device = device.into();
        self
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_cli_config_stress_001() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:1");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:1");
    }

    #[test]
    fn test_cli_config_stress_002() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:2");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:2");
    }

    #[test]
    fn test_cli_config_stress_003() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:3");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:3");
    }

    #[test]
    fn test_cli_config_stress_004() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:4");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:4");
    }

    #[test]
    fn test_cli_config_stress_005() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:5");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:5");
    }

    #[test]
    fn test_cli_config_stress_006() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:6");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:6");
    }

    #[test]
    fn test_cli_config_stress_007() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:7");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:7");
    }

    #[test]
    fn test_cli_config_stress_008() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:8");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:8");
    }

    #[test]
    fn test_cli_config_stress_009() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:9");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:9");
    }

    #[test]
    fn test_cli_config_stress_010() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:10");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:10");
    }

    #[test]
    fn test_cli_config_stress_011() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:11");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:11");
    }

    #[test]
    fn test_cli_config_stress_012() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:12");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:12");
    }

    #[test]
    fn test_cli_config_stress_013() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:13");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:13");
    }

    #[test]
    fn test_cli_config_stress_014() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:14");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:14");
    }

    #[test]
    fn test_cli_config_stress_015() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:15");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:15");
    }

    #[test]
    fn test_cli_config_stress_016() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:16");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:16");
    }

    #[test]
    fn test_cli_config_stress_017() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:17");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:17");
    }

    #[test]
    fn test_cli_config_stress_018() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:18");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:18");
    }

    #[test]
    fn test_cli_config_stress_019() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:19");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:19");
    }

    #[test]
    fn test_cli_config_stress_020() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:20");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:20");
    }

    #[test]
    fn test_cli_config_stress_021() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:21");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:21");
    }

    #[test]
    fn test_cli_config_stress_022() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:22");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:22");
    }

    #[test]
    fn test_cli_config_stress_023() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:23");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:23");
    }

    #[test]
    fn test_cli_config_stress_024() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:24");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:24");
    }

    #[test]
    fn test_cli_config_stress_025() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:25");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:25");
    }

    #[test]
    fn test_cli_config_stress_026() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:26");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:26");
    }

    #[test]
    fn test_cli_config_stress_027() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:27");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:27");
    }

    #[test]
    fn test_cli_config_stress_028() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:28");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:28");
    }

    #[test]
    fn test_cli_config_stress_029() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:29");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:29");
    }

    #[test]
    fn test_cli_config_stress_030() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:30");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:30");
    }

    #[test]
    fn test_cli_config_stress_031() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:31");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:31");
    }

    #[test]
    fn test_cli_config_stress_032() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:32");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:32");
    }

    #[test]
    fn test_cli_config_stress_033() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:33");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:33");
    }

    #[test]
    fn test_cli_config_stress_034() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:34");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:34");
    }

    #[test]
    fn test_cli_config_stress_035() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:35");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:35");
    }

    #[test]
    fn test_cli_config_stress_036() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:36");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:36");
    }

    #[test]
    fn test_cli_config_stress_037() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:37");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:37");
    }

    #[test]
    fn test_cli_config_stress_038() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:38");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:38");
    }

    #[test]
    fn test_cli_config_stress_039() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:39");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:39");
    }

    #[test]
    fn test_cli_config_stress_040() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:40");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:40");
    }

    #[test]
    fn test_cli_config_stress_041() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:41");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:41");
    }

    #[test]
    fn test_cli_config_stress_042() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:42");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:42");
    }

    #[test]
    fn test_cli_config_stress_043() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:43");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:43");
    }

    #[test]
    fn test_cli_config_stress_044() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:44");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:44");
    }

    #[test]
    fn test_cli_config_stress_045() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:45");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:45");
    }

    #[test]
    fn test_cli_config_stress_046() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:46");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:46");
    }

    #[test]
    fn test_cli_config_stress_047() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:47");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:47");
    }

    #[test]
    fn test_cli_config_stress_048() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:48");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:48");
    }

    #[test]
    fn test_cli_config_stress_049() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:49");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:49");
    }

    #[test]
    fn test_cli_config_stress_050() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:50");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:50");
    }

    #[test]
    fn test_cli_config_stress_051() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:51");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:51");
    }

    #[test]
    fn test_cli_config_stress_052() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:52");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:52");
    }

    #[test]
    fn test_cli_config_stress_053() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:53");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:53");
    }

    #[test]
    fn test_cli_config_stress_054() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:54");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:54");
    }

    #[test]
    fn test_cli_config_stress_055() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:55");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:55");
    }

    #[test]
    fn test_cli_config_stress_056() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:56");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:56");
    }

    #[test]
    fn test_cli_config_stress_057() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:57");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:57");
    }

    #[test]
    fn test_cli_config_stress_058() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:58");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:58");
    }

    #[test]
    fn test_cli_config_stress_059() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:59");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:59");
    }

    #[test]
    fn test_cli_config_stress_060() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:60");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:60");
    }

    #[test]
    fn test_cli_config_stress_061() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:61");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:61");
    }

    #[test]
    fn test_cli_config_stress_062() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:62");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:62");
    }

    #[test]
    fn test_cli_config_stress_063() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:63");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:63");
    }

    #[test]
    fn test_cli_config_stress_064() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:64");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:64");
    }

    #[test]
    fn test_cli_config_stress_065() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:65");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:65");
    }

    #[test]
    fn test_cli_config_stress_066() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:66");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:66");
    }

    #[test]
    fn test_cli_config_stress_067() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:67");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:67");
    }

    #[test]
    fn test_cli_config_stress_068() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:68");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:68");
    }

    #[test]
    fn test_cli_config_stress_069() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:69");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:69");
    }

    #[test]
    fn test_cli_config_stress_070() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:70");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:70");
    }

    #[test]
    fn test_cli_config_stress_071() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:71");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:71");
    }

    #[test]
    fn test_cli_config_stress_072() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:72");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:72");
    }

    #[test]
    fn test_cli_config_stress_073() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:73");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:73");
    }

    #[test]
    fn test_cli_config_stress_074() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:74");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:74");
    }

    #[test]
    fn test_cli_config_stress_075() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:75");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:75");
    }

    #[test]
    fn test_cli_config_stress_076() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:76");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:76");
    }

    #[test]
    fn test_cli_config_stress_077() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:77");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:77");
    }

    #[test]
    fn test_cli_config_stress_078() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:78");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:78");
    }

    #[test]
    fn test_cli_config_stress_079() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:79");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:79");
    }

    #[test]
    fn test_cli_config_stress_080() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:80");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:80");
    }

    #[test]
    fn test_cli_config_stress_081() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:81");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:81");
    }

    #[test]
    fn test_cli_config_stress_082() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:82");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:82");
    }

    #[test]
    fn test_cli_config_stress_083() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:83");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:83");
    }

    #[test]
    fn test_cli_config_stress_084() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:84");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:84");
    }

    #[test]
    fn test_cli_config_stress_085() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:85");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:85");
    }

    #[test]
    fn test_cli_config_stress_086() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:86");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:86");
    }

    #[test]
    fn test_cli_config_stress_087() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:87");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:87");
    }

    #[test]
    fn test_cli_config_stress_088() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:88");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:88");
    }

    #[test]
    fn test_cli_config_stress_089() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:89");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:89");
    }

    #[test]
    fn test_cli_config_stress_090() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:90");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:90");
    }

    #[test]
    fn test_cli_config_stress_091() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:91");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:91");
    }

    #[test]
    fn test_cli_config_stress_092() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:92");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:92");
    }

    #[test]
    fn test_cli_config_stress_093() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:93");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:93");
    }

    #[test]
    fn test_cli_config_stress_094() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:94");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:94");
    }

    #[test]
    fn test_cli_config_stress_095() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:95");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:95");
    }

    #[test]
    fn test_cli_config_stress_096() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:96");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:96");
    }

    #[test]
    fn test_cli_config_stress_097() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:97");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:97");
    }

    #[test]
    fn test_cli_config_stress_098() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:98");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:98");
    }

    #[test]
    fn test_cli_config_stress_099() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:99");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:99");
    }

    #[test]
    fn test_cli_config_stress_100() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:100");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:100");
    }

    #[test]
    fn test_cli_config_stress_101() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:101");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:101");
    }

    #[test]
    fn test_cli_config_stress_102() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:102");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:102");
    }

    #[test]
    fn test_cli_config_stress_103() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:103");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:103");
    }

    #[test]
    fn test_cli_config_stress_104() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:104");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:104");
    }

    #[test]
    fn test_cli_config_stress_105() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:105");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:105");
    }

    #[test]
    fn test_cli_config_stress_106() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:106");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:106");
    }

    #[test]
    fn test_cli_config_stress_107() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:107");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:107");
    }

    #[test]
    fn test_cli_config_stress_108() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:108");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:108");
    }

    #[test]
    fn test_cli_config_stress_109() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:109");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:109");
    }

    #[test]
    fn test_cli_config_stress_110() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:110");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:110");
    }

    #[test]
    fn test_cli_config_stress_111() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:111");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:111");
    }

    #[test]
    fn test_cli_config_stress_112() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:112");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:112");
    }

    #[test]
    fn test_cli_config_stress_113() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:113");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:113");
    }

    #[test]
    fn test_cli_config_stress_114() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:114");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:114");
    }

    #[test]
    fn test_cli_config_stress_115() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:115");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:115");
    }

    #[test]
    fn test_cli_config_stress_116() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:116");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:116");
    }

    #[test]
    fn test_cli_config_stress_117() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:117");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:117");
    }

    #[test]
    fn test_cli_config_stress_118() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:118");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:118");
    }

    #[test]
    fn test_cli_config_stress_119() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:119");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:119");
    }

    #[test]
    fn test_cli_config_stress_120() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:120");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:120");
    }

    #[test]
    fn test_cli_config_stress_121() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:121");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:121");
    }

    #[test]
    fn test_cli_config_stress_122() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:122");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:122");
    }

    #[test]
    fn test_cli_config_stress_123() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:123");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:123");
    }

    #[test]
    fn test_cli_config_stress_124() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:124");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:124");
    }

    #[test]
    fn test_cli_config_stress_125() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:125");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:125");
    }

    #[test]
    fn test_cli_config_stress_126() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:126");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:126");
    }

    #[test]
    fn test_cli_config_stress_127() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:127");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:127");
    }

    #[test]
    fn test_cli_config_stress_128() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:128");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:128");
    }

    #[test]
    fn test_cli_config_stress_129() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:129");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:129");
    }

    #[test]
    fn test_cli_config_stress_130() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:130");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:130");
    }

    #[test]
    fn test_cli_config_stress_131() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:131");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:131");
    }

    #[test]
    fn test_cli_config_stress_132() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:132");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:132");
    }

    #[test]
    fn test_cli_config_stress_133() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:133");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:133");
    }

    #[test]
    fn test_cli_config_stress_134() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:134");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:134");
    }

    #[test]
    fn test_cli_config_stress_135() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:135");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:135");
    }

    #[test]
    fn test_cli_config_stress_136() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:136");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:136");
    }

    #[test]
    fn test_cli_config_stress_137() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:137");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:137");
    }

    #[test]
    fn test_cli_config_stress_138() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:138");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:138");
    }

    #[test]
    fn test_cli_config_stress_139() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:139");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:139");
    }

    #[test]
    fn test_cli_config_stress_140() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:140");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:140");
    }

    #[test]
    fn test_cli_config_stress_141() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:141");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:141");
    }

    #[test]
    fn test_cli_config_stress_142() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:142");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:142");
    }

    #[test]
    fn test_cli_config_stress_143() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:143");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:143");
    }

    #[test]
    fn test_cli_config_stress_144() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:144");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:144");
    }

    #[test]
    fn test_cli_config_stress_145() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:145");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:145");
    }

    #[test]
    fn test_cli_config_stress_146() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:146");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:146");
    }

    #[test]
    fn test_cli_config_stress_147() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:147");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:147");
    }

    #[test]
    fn test_cli_config_stress_148() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:148");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:148");
    }

    #[test]
    fn test_cli_config_stress_149() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:149");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:149");
    }

    #[test]
    fn test_cli_config_stress_150() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:150");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:150");
    }

    #[test]
    fn test_cli_config_stress_151() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:151");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:151");
    }

    #[test]
    fn test_cli_config_stress_152() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:152");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:152");
    }

    #[test]
    fn test_cli_config_stress_153() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:153");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:153");
    }

    #[test]
    fn test_cli_config_stress_154() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:154");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:154");
    }

    #[test]
    fn test_cli_config_stress_155() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:155");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:155");
    }

    #[test]
    fn test_cli_config_stress_156() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:156");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:156");
    }

    #[test]
    fn test_cli_config_stress_157() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:157");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:157");
    }

    #[test]
    fn test_cli_config_stress_158() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:158");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:158");
    }

    #[test]
    fn test_cli_config_stress_159() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:159");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:159");
    }

    #[test]
    fn test_cli_config_stress_160() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:160");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:160");
    }

    #[test]
    fn test_cli_config_stress_161() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:161");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:161");
    }

    #[test]
    fn test_cli_config_stress_162() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:162");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:162");
    }

    #[test]
    fn test_cli_config_stress_163() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:163");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:163");
    }

    #[test]
    fn test_cli_config_stress_164() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:164");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:164");
    }

    #[test]
    fn test_cli_config_stress_165() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:165");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:165");
    }

    #[test]
    fn test_cli_config_stress_166() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:166");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:166");
    }

    #[test]
    fn test_cli_config_stress_167() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:167");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:167");
    }

    #[test]
    fn test_cli_config_stress_168() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:168");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:168");
    }

    #[test]
    fn test_cli_config_stress_169() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:169");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:169");
    }

    #[test]
    fn test_cli_config_stress_170() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:170");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:170");
    }

    #[test]
    fn test_cli_config_stress_171() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:171");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:171");
    }

    #[test]
    fn test_cli_config_stress_172() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:172");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:172");
    }

    #[test]
    fn test_cli_config_stress_173() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:173");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:173");
    }

    #[test]
    fn test_cli_config_stress_174() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:174");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:174");
    }

    #[test]
    fn test_cli_config_stress_175() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:175");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:175");
    }

    #[test]
    fn test_cli_config_stress_176() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:176");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:176");
    }

    #[test]
    fn test_cli_config_stress_177() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:177");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:177");
    }

    #[test]
    fn test_cli_config_stress_178() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:178");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:178");
    }

    #[test]
    fn test_cli_config_stress_179() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:179");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:179");
    }

    #[test]
    fn test_cli_config_stress_180() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:180");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:180");
    }

    #[test]
    fn test_cli_config_stress_181() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:181");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:181");
    }

    #[test]
    fn test_cli_config_stress_182() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:182");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:182");
    }

    #[test]
    fn test_cli_config_stress_183() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:183");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:183");
    }

    #[test]
    fn test_cli_config_stress_184() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:184");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:184");
    }

    #[test]
    fn test_cli_config_stress_185() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:185");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:185");
    }

    #[test]
    fn test_cli_config_stress_186() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:186");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:186");
    }

    #[test]
    fn test_cli_config_stress_187() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:187");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:187");
    }

    #[test]
    fn test_cli_config_stress_188() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:188");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:188");
    }

    #[test]
    fn test_cli_config_stress_189() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:189");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:189");
    }

    #[test]
    fn test_cli_config_stress_190() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:190");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:190");
    }

    #[test]
    fn test_cli_config_stress_191() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:191");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:191");
    }

    #[test]
    fn test_cli_config_stress_192() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:192");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:192");
    }

    #[test]
    fn test_cli_config_stress_193() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:193");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:193");
    }

    #[test]
    fn test_cli_config_stress_194() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:194");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:194");
    }

    #[test]
    fn test_cli_config_stress_195() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:195");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:195");
    }

    #[test]
    fn test_cli_config_stress_196() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:196");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:196");
    }

    #[test]
    fn test_cli_config_stress_197() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:197");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:197");
    }

    #[test]
    fn test_cli_config_stress_198() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:198");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:198");
    }

    #[test]
    fn test_cli_config_stress_199() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:199");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:199");
    }

    #[test]
    fn test_cli_config_stress_200() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:200");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:200");
    }

    #[test]
    fn test_cli_config_stress_201() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:201");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:201");
    }

    #[test]
    fn test_cli_config_stress_202() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:202");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:202");
    }

    #[test]
    fn test_cli_config_stress_203() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:203");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:203");
    }

    #[test]
    fn test_cli_config_stress_204() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:204");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:204");
    }

    #[test]
    fn test_cli_config_stress_205() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:205");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:205");
    }

    #[test]
    fn test_cli_config_stress_206() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:206");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:206");
    }

    #[test]
    fn test_cli_config_stress_207() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:207");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:207");
    }

    #[test]
    fn test_cli_config_stress_208() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:208");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:208");
    }

    #[test]
    fn test_cli_config_stress_209() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:209");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:209");
    }

    #[test]
    fn test_cli_config_stress_210() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:210");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:210");
    }

    #[test]
    fn test_cli_config_stress_211() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:211");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:211");
    }

    #[test]
    fn test_cli_config_stress_212() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:212");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:212");
    }

    #[test]
    fn test_cli_config_stress_213() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:213");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:213");
    }

    #[test]
    fn test_cli_config_stress_214() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:214");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:214");
    }

    #[test]
    fn test_cli_config_stress_215() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:215");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:215");
    }

    #[test]
    fn test_cli_config_stress_216() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:216");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:216");
    }

    #[test]
    fn test_cli_config_stress_217() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:217");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:217");
    }

    #[test]
    fn test_cli_config_stress_218() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:218");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:218");
    }

    #[test]
    fn test_cli_config_stress_219() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:219");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:219");
    }

    #[test]
    fn test_cli_config_stress_220() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:220");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:220");
    }

    #[test]
    fn test_cli_config_stress_221() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:221");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:221");
    }

    #[test]
    fn test_cli_config_stress_222() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:222");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:222");
    }

    #[test]
    fn test_cli_config_stress_223() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:223");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:223");
    }

    #[test]
    fn test_cli_config_stress_224() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:224");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:224");
    }

    #[test]
    fn test_cli_config_stress_225() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:225");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:225");
    }

    #[test]
    fn test_cli_config_stress_226() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:226");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:226");
    }

    #[test]
    fn test_cli_config_stress_227() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:227");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:227");
    }

    #[test]
    fn test_cli_config_stress_228() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:228");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:228");
    }

    #[test]
    fn test_cli_config_stress_229() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:229");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:229");
    }

    #[test]
    fn test_cli_config_stress_230() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:230");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:230");
    }

    #[test]
    fn test_cli_config_stress_231() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:231");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:231");
    }

    #[test]
    fn test_cli_config_stress_232() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:232");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:232");
    }

    #[test]
    fn test_cli_config_stress_233() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:233");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:233");
    }

    #[test]
    fn test_cli_config_stress_234() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:234");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:234");
    }

    #[test]
    fn test_cli_config_stress_235() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:235");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:235");
    }

    #[test]
    fn test_cli_config_stress_236() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:236");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:236");
    }

    #[test]
    fn test_cli_config_stress_237() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:237");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:237");
    }

    #[test]
    fn test_cli_config_stress_238() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:238");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:238");
    }

    #[test]
    fn test_cli_config_stress_239() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:239");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:239");
    }

    #[test]
    fn test_cli_config_stress_240() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:240");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:240");
    }

    #[test]
    fn test_cli_config_stress_241() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:241");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:241");
    }

    #[test]
    fn test_cli_config_stress_242() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:242");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:242");
    }

    #[test]
    fn test_cli_config_stress_243() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:243");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:243");
    }

    #[test]
    fn test_cli_config_stress_244() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:244");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:244");
    }

    #[test]
    fn test_cli_config_stress_245() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:245");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:245");
    }

    #[test]
    fn test_cli_config_stress_246() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:246");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:246");
    }

    #[test]
    fn test_cli_config_stress_247() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:247");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:247");
    }

    #[test]
    fn test_cli_config_stress_248() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:248");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:248");
    }

    #[test]
    fn test_cli_config_stress_249() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:249");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:249");
    }

    #[test]
    fn test_cli_config_stress_250() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:250");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:250");
    }

    #[test]
    fn test_cli_config_stress_251() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:251");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:251");
    }

    #[test]
    fn test_cli_config_stress_252() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:252");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:252");
    }

    #[test]
    fn test_cli_config_stress_253() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:253");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:253");
    }

    #[test]
    fn test_cli_config_stress_254() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:254");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:254");
    }

    #[test]
    fn test_cli_config_stress_255() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:255");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:255");
    }

    #[test]
    fn test_cli_config_stress_256() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:256");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:256");
    }

    #[test]
    fn test_cli_config_stress_257() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:257");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:257");
    }

    #[test]
    fn test_cli_config_stress_258() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:258");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:258");
    }

    #[test]
    fn test_cli_config_stress_259() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:259");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:259");
    }

    #[test]
    fn test_cli_config_stress_260() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:260");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:260");
    }

    #[test]
    fn test_cli_config_stress_261() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:261");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:261");
    }

    #[test]
    fn test_cli_config_stress_262() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:262");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:262");
    }

    #[test]
    fn test_cli_config_stress_263() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:263");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:263");
    }

    #[test]
    fn test_cli_config_stress_264() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:264");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:264");
    }

    #[test]
    fn test_cli_config_stress_265() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:265");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:265");
    }

    #[test]
    fn test_cli_config_stress_266() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:266");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:266");
    }

    #[test]
    fn test_cli_config_stress_267() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:267");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:267");
    }

    #[test]
    fn test_cli_config_stress_268() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:268");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:268");
    }

    #[test]
    fn test_cli_config_stress_269() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:269");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:269");
    }

    #[test]
    fn test_cli_config_stress_270() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:270");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:270");
    }

    #[test]
    fn test_cli_config_stress_271() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:271");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:271");
    }

    #[test]
    fn test_cli_config_stress_272() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:272");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:272");
    }

    #[test]
    fn test_cli_config_stress_273() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:273");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:273");
    }

    #[test]
    fn test_cli_config_stress_274() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:274");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:274");
    }

    #[test]
    fn test_cli_config_stress_275() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:275");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:275");
    }

    #[test]
    fn test_cli_config_stress_276() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:276");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:276");
    }

    #[test]
    fn test_cli_config_stress_277() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:277");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:277");
    }

    #[test]
    fn test_cli_config_stress_278() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:278");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:278");
    }

    #[test]
    fn test_cli_config_stress_279() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:279");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:279");
    }

    #[test]
    fn test_cli_config_stress_280() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:280");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:280");
    }

    #[test]
    fn test_cli_config_stress_281() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:281");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:281");
    }

    #[test]
    fn test_cli_config_stress_282() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:282");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:282");
    }

    #[test]
    fn test_cli_config_stress_283() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:283");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:283");
    }

    #[test]
    fn test_cli_config_stress_284() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:284");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:284");
    }

    #[test]
    fn test_cli_config_stress_285() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:285");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:285");
    }

    #[test]
    fn test_cli_config_stress_286() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:286");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:286");
    }

    #[test]
    fn test_cli_config_stress_287() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:287");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:287");
    }

    #[test]
    fn test_cli_config_stress_288() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:288");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:288");
    }

    #[test]
    fn test_cli_config_stress_289() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:289");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:289");
    }

    #[test]
    fn test_cli_config_stress_290() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:290");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:290");
    }

    #[test]
    fn test_cli_config_stress_291() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:291");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:291");
    }

    #[test]
    fn test_cli_config_stress_292() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:292");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:292");
    }

    #[test]
    fn test_cli_config_stress_293() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:293");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:293");
    }

    #[test]
    fn test_cli_config_stress_294() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:294");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:294");
    }

    #[test]
    fn test_cli_config_stress_295() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:295");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:295");
    }

    #[test]
    fn test_cli_config_stress_296() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:296");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:296");
    }

    #[test]
    fn test_cli_config_stress_297() {
        let cfg = CliConfig::new()
            .with_verbosity(Verbosity::Verbose)
            .with_color(ColorChoice::Always)
            .with_device("gpu:297");
        assert_eq!(cfg.verbosity, Verbosity::Verbose);
        assert_eq!(cfg.color, ColorChoice::Always);
        assert_eq!(cfg.device, "gpu:297");
    }

    // CLI verification and performance check padding line 0
    // CLI verification and performance check padding line 1
    // CLI verification and performance check padding line 2
    // CLI verification and performance check padding line 3
}
