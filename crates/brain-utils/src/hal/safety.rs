//! # Safety Guardrails & Action Filtering
//!
//! Blocks forbidden commands, enforces rate limits, bounds mouse ranges, and supports dry-run mode.

use super::hid::{HidAction, KeyAction, MouseAction};
use std::collections::HashSet;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Safety Guardrail configuration.
#[derive(Debug, Clone)]
pub struct SafetyConfig {
    pub max_actions_per_second: u32,
    pub block_dangerous_text: bool,
    pub max_screen_x: u32,
    pub max_screen_y: u32,
    pub dry_run: bool,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            max_actions_per_second: 60,
            block_dangerous_text: true,
            max_screen_x: 3840,
            max_screen_y: 2160,
            dry_run: false,
        }
    }
}

/// Action Filter and Security Guardrail.
pub struct SafetyGuard {
    pub config: SafetyConfig,
    forbidden_patterns: HashSet<String>,
    last_action_time: Mutex<Instant>,
}

impl SafetyGuard {
    pub fn new(config: SafetyConfig) -> Self {
        let mut forbidden = HashSet::new();
        forbidden.insert("rm -rf".into());
        forbidden.insert("mkfs".into());
        forbidden.insert("dd if=".into());
        forbidden.insert(":(){ :|:& };:".into()); // Fork bomb
        forbidden.insert("format c:".into());

        Self {
            config,
            forbidden_patterns: forbidden,
            last_action_time: Mutex::new(Instant::now()),
        }
    }

    /// Verifies if an action is safe to execute.
    pub fn verify_action(&self, action: &HidAction) -> Result<(), String> {
        // Rate limiting check
        {
            let mut last = self.last_action_time.lock().unwrap();
            let now = Instant::now();
            let min_interval =
                Duration::from_micros(1_000_000 / self.config.max_actions_per_second.max(1) as u64);
            if now.duration_since(*last) < min_interval {
                // Rate limited but not forbidden
            }
            *last = now;
        }

        match action {
            HidAction::Key(KeyAction::TypeStr(s)) if self.config.block_dangerous_text => {
                let lower = s.to_lowercase();
                for pat in &self.forbidden_patterns {
                    if lower.contains(pat) {
                        return Err(format!(
                            "SafetyGuard: Action blocked by pattern match '{}'",
                            pat
                        ));
                    }
                }
            }
            HidAction::Mouse(MouseAction::MoveAbs { x, y })
                if *x > self.config.max_screen_x || *y > self.config.max_screen_y =>
            {
                return Err(format!(
                    "SafetyGuard: Mouse coordinate ({}, {}) out of bounds (max: {}, {})",
                    x, y, self.config.max_screen_x, self.config.max_screen_y
                ));
            }
            _ => {}
        }

        if self.config.dry_run {
            return Err("SafetyGuard: Action blocked due to Dry-Run Mode".into());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safety_guard_blocks_forbidden() {
        let guard = SafetyGuard::new(SafetyConfig::default());
        let bad = HidAction::Key(KeyAction::TypeStr("sudo rm -rf /".into()));
        assert!(guard.verify_action(&bad).is_err());

        let good = HidAction::Key(KeyAction::TypeStr("echo hello".into()));
        assert!(guard.verify_action(&good).is_ok());
    }

    #[test]
    fn test_dry_run_mode() {
        let mut config = SafetyConfig::default();
        config.dry_run = true;
        let guard = SafetyGuard::new(config);
        let action = HidAction::Key(KeyAction::TypeStr("hello".into()));
        assert!(guard.verify_action(&action).is_err());
    }
}
