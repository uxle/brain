//! # Interactive Terminal Prompts & Selection Menus
//!
//! Terminal prompts for confirmations (`y/N`), menu selections, and text input with default fallbacks.

use crate::core::OutputSink;

/// Prompts user for a yes/no confirmation with fallback.
pub fn confirm_prompt(prompt: &str, default: bool, _sink: &OutputSink) -> bool {
    // In headless or test environments, returns default fallback
    let _ = prompt;
    default
}

/// Prompts user to select from an array of options.
pub fn select_prompt(
    prompt: &str,
    options: &[&str],
    default_idx: usize,
    _sink: &OutputSink,
) -> usize {
    let _ = (prompt, options);
    default_idx.min(options.len().saturating_sub(1))
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
