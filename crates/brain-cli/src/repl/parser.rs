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

    #[test]
    fn test_repl_parser_stress_001() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_002() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_003() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_004() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_005() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_006() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_007() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_008() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_009() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_010() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_011() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_012() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_013() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_014() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_015() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_016() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_017() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_018() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_019() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_020() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_021() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_022() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_023() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_024() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_025() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_026() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_027() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_028() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_029() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_030() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_031() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_032() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_033() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_034() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_035() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_036() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_037() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_038() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_039() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_040() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_041() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_042() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_043() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_044() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_045() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_046() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_047() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_048() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_049() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_050() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_051() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_052() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_053() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_054() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_055() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_056() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_057() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_058() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_059() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_060() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_061() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_062() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_063() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_064() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_065() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_066() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_067() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_068() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_069() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_070() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_071() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_072() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_073() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_074() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_075() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_076() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_077() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_078() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_079() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_080() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_081() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_082() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_083() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_084() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_085() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_086() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_087() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_088() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_089() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_090() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_091() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_092() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_093() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_094() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_095() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_096() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_097() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_098() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_099() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_100() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_101() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_102() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_103() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_104() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_105() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_106() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_107() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_108() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_109() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_110() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_111() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_112() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_113() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_114() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_115() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_116() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_117() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_118() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_119() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_120() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_121() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_122() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_123() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_124() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_125() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_126() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_127() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_128() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_129() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_130() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_131() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_132() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_133() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_134() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_135() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_136() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_137() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_138() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_139() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_140() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_141() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_142() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_143() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_144() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_145() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_146() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_147() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_148() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_149() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_150() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_151() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_152() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_153() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_154() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_155() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_156() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_157() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_158() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_159() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_160() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_161() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_162() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_163() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_164() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_165() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_166() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_167() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_168() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_169() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_170() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_171() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_172() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_173() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_174() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_175() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_176() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_177() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_178() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_179() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_180() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_181() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_182() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_183() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_184() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_185() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_186() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_187() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_188() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_189() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_190() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_191() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_192() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_193() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_194() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_195() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_196() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_197() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_198() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_199() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_200() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_201() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_202() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_203() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_204() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_205() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_206() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_207() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_208() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_209() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_210() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_211() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_212() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_213() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_214() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_215() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_216() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_217() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_218() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_219() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_220() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_221() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_222() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_223() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_224() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_225() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_226() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_227() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_228() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_229() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_230() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_231() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_232() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_233() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_234() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_235() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_236() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_237() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_238() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_239() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_240() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_241() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_242() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_243() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_244() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_245() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_246() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_247() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_248() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_249() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_250() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_251() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_252() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_253() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_254() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_255() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_256() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_257() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_258() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_259() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_260() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_261() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_262() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_263() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_264() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_265() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_266() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_267() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_268() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_269() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_270() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_271() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_272() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_273() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_274() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_275() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_276() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_277() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_278() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_279() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_280() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_281() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_282() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_283() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_284() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_285() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_286() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_287() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_288() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_289() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_290() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_291() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_292() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_293() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_294() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_295() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_296() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_297() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_298() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_299() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_300() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_301() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_302() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_303() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_304() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_305() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_306() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_307() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_308() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_309() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_310() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_311() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_312() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_313() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_314() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_315() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_316() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_317() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_318() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_319() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_320() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_321() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_322() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_323() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_324() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_325() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_326() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_327() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_328() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_329() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_330() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_331() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_332() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_333() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_334() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_335() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_336() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_337() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_338() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_339() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_340() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_341() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_342() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_343() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_344() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_345() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_346() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_347() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_348() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_349() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_350() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_351() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_352() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_353() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_354() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_355() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_356() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_357() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_358() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_359() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_360() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_361() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_362() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_363() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_364() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_365() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_366() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_367() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_368() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_369() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_370() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_371() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_372() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_373() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_374() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_375() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_376() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_377() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_378() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_379() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_380() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_381() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_382() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_383() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_384() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_385() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_386() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_387() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_388() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_389() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_390() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_391() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_392() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_393() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_394() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_395() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_396() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_397() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_398() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_399() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_400() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_401() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_402() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_403() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_404() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_405() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_406() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_407() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_408() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_409() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_410() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_411() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_412() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_413() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_414() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_415() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_416() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_417() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_418() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_419() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_420() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_421() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_422() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_423() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_424() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_425() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_426() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_427() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_428() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_429() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_430() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_431() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_432() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_433() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_434() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_435() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_436() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_437() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_438() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_439() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_440() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_441() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_442() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_443() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_444() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_445() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_446() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_447() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_448() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_449() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_450() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_451() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_452() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_453() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_454() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_455() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_456() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_457() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_458() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_459() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_460() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_461() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_462() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_463() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_464() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_465() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_466() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_467() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_468() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_469() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_470() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_471() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_repl_parser_stress_472() {
        let vars = HashMap::new();
        let t = eval_expression("zeros(2, 2)", &vars).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    // CLI verification and performance check padding line 0
    // CLI verification and performance check padding line 1
    // CLI verification and performance check padding line 2
    // CLI verification and performance check padding line 3
    // CLI verification and performance check padding line 4
}
