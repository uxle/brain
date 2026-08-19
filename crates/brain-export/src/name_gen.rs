//! # Deterministic Name Sanitization & Generation
//!
//! Generates unique identifiers adhering to format naming constraints (e.g. C-identifiers).

/// Sanitizes a string into a valid C-style identifier.
pub fn sanitize_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for c in name.chars() {
        if c.is_alphanumeric() || c == '_' {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() || out.chars().next().unwrap().is_numeric() {
        out.insert(0, '_');
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
