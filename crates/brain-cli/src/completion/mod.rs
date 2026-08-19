//! # Shell Completion Script Generators
//!
//! Generates auto-completion scripts for Bash, Zsh, Fish, and PowerShell shells.

/// Shell type target for completion script synthesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shell {
    Bash,
    Zsh,
    Fish,
    PowerShell,
}

/// Generates shell completion script for the target shell.
pub fn generate_completion_script(shell: Shell) -> String {
    match shell {
        Shell::Bash => {
            let mut s = String::new();
            s.push_str("_brain_completions() {\n");
            s.push_str("    COMPREPLY=( $(compgen -W 'tensor model train bench dataset convert repl doctor init' -- \"$2\") )\n");
            s.push_str("}\n");
            s.push_str("complete -F _brain_completions brain\n");
            s
        }
        Shell::Zsh => {
            let mut s = String::new();
            s.push_str("#compdef brain\n");
            s.push_str("_brain() {\n");
            s.push_str("    _arguments '1: :((tensor model train bench dataset convert repl doctor init))'\n");
            s.push_str("}\n");
            s
        }
        Shell::Fish => {
            "complete -c brain -f -a 'tensor model train bench dataset convert repl doctor init'\n".to_string()
        }
        Shell::PowerShell => {
            "Register-ArgumentCompleter -Native -CommandName brain\n".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
