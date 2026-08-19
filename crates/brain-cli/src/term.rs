//! # Terminal ANSI Styling, Progress Bars & Spinners
//!
//! Terminal output utilities including ANSI color styles, dynamic progress bars, and spinners.

/// ANSI Color and Text Styles.
pub struct Style;

impl Style {
    pub const RESET: &'static str = "\x1b[0m";
    pub const BOLD: &'static str = "\x1b[1m";
    pub const RED: &'static str = "\x1b[31m";
    pub const GREEN: &'static str = "\x1b[32m";
    pub const YELLOW: &'static str = "\x1b[33m";
    pub const BLUE: &'static str = "\x1b[34m";
    pub const CYAN: &'static str = "\x1b[36m";

    /// Wraps text in green color styling.
    pub fn green(s: &str) -> String {
        format!("{}{}{}", Self::GREEN, s, Self::RESET)
    }

    /// Wraps text in red color styling.
    pub fn red(s: &str) -> String {
        format!("{}{}{}", Self::RED, s, Self::RESET)
    }

    /// Wraps text in bold styling.
    pub fn bold(s: &str) -> String {
        format!("{}{}{}", Self::BOLD, s, Self::RESET)
    }
}

/// Dynamic Progress Bar for CLI long-running tasks.
#[derive(Debug, Clone)]
pub struct ProgressBar {
    total: usize,
    current: usize,
    width: usize,
}

impl ProgressBar {
    /// Creates a new progress bar with `total` steps.
    pub fn new(total: usize) -> Self {
        Self {
            total: total.max(1),
            current: 0,
            width: 30,
        }
    }

    /// Advances the current step count.
    pub fn inc(&mut self, delta: usize) {
        self.current = (self.current + delta).min(self.total);
    }

    /// Formats the progress bar into a display string.
    pub fn render(&self) -> String {
        let pct = (self.current as f64 / self.total as f64).clamp(0.0, 1.0);
        let filled = (pct * self.width as f64).round() as usize;
        let empty = self.width.saturating_sub(filled);

        let bar = format!("[{}{}]", "=".repeat(filled), " ".repeat(empty));
        format!("{} {:>3.0}% ({}/{})", bar, pct * 100.0, self.current, self.total)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
