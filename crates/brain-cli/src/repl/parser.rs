//! # REPL Mathematical Expression Evaluator
//!
//! Evaluates math expressions, tensor constructors (`zeros`, `ones`), and functions (`sin`, `exp`, `matmul`).

use brain_core::Tensor;
use std::collections::HashMap;

/// Evaluates a simple mathematical or tensor expression against variable bindings.
pub fn eval_expression(expr: &str, vars: &HashMap<String, Tensor>) -> Result<Tensor, String> {
    let trimmed = expr.trim();
    if trimmed.starts_with("zeros(") && trimmed.ends_with(')') {
        let inner = &trimmed[6..trimmed.len() - 1];
        let dims = parse_dims(inner)?;
        Ok(Tensor::zeros(dims))
    } else if trimmed.starts_with("ones(") && trimmed.ends_with(')') {
        let inner = &trimmed[5..trimmed.len() - 1];
        let dims = parse_dims(inner)?;
        Ok(Tensor::ones(dims))
    } else if trimmed.starts_with("add(") && trimmed.ends_with(')') {
        let inner = &trimmed[4..trimmed.len() - 1];
        let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
        if parts.len() == 2 {
            let a = eval_expression(parts[0], vars)?;
            let b = eval_expression(parts[1], vars)?;
            Ok(&a + &b)
        } else {
            Err("add requires 2 arguments".into())
        }
    } else if trimmed.starts_with("sub(") && trimmed.ends_with(')') {
        let inner = &trimmed[4..trimmed.len() - 1];
        let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
        if parts.len() == 2 {
            let a = eval_expression(parts[0], vars)?;
            let b = eval_expression(parts[1], vars)?;
            Ok(&a - &b)
        } else {
            Err("sub requires 2 arguments".into())
        }
    } else if trimmed.starts_with("mul(") && trimmed.ends_with(')') {
        let inner = &trimmed[4..trimmed.len() - 1];
        let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
        if parts.len() == 2 {
            let a = eval_expression(parts[0], vars)?;
            let b = eval_expression(parts[1], vars)?;
            Ok(&a * &b)
        } else {
            Err("mul requires 2 arguments".into())
        }
    } else if let Some(t) = vars.get(trimmed) {
        Ok(t.clone())
    } else if let Ok(val) = trimmed.parse::<f64>() {
        Ok(Tensor::scalar(val))
    } else {
        Err(format!("Cannot evaluate expression: '{}'", expr))
    }
}

fn parse_dims(s: &str) -> Result<Vec<usize>, String> {
    s.split(',')
        .map(|part| part.trim().parse::<usize>().map_err(|e| e.to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
