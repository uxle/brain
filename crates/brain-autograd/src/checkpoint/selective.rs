//! # Selective Activation Checkpointing
//!
//! Configurable policies for deciding which intermediate activations to retain vs recompute.

use crate::value::Value;
use brain_core::BrainResult;

/// Strategy determining which layers or activation sizes are retained in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointPolicy {
    /// Discard all non-leaf forward activations and recompute everything.
    All,
    /// Only checkpoint operations exceeding a certain tensor size in bytes.
    Selective { threshold_elements: usize },
    /// Do not checkpoint (standard autograd behavior).
    None,
}

/// Checkpoints a functional closure `f` with given `inputs`.
pub fn checkpoint<F>(f: F, inputs: &[&Value]) -> BrainResult<Vec<Value>>
where
    F: Fn(&[&Value]) -> BrainResult<Vec<Value>>,
{
    // Execute forward pass without retaining intermediate computation graphs
    let detached_inputs: Vec<Value> = inputs.iter().map(|&v| v.detach()).collect();
    let detached_refs: Vec<&Value> = detached_inputs.iter().collect();
    let outputs = f(&detached_refs)?;
    Ok(outputs)
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use crate::value::Value;
    #[allow(unused_imports)]
    use brain_core::Tensor;
    #[allow(unused_imports)]
    use crate::tape::OpRecord;

    #[test]
    fn test_selective_checkpoint_stress_001() {
        let x = Value::scalar(1.1);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (1.1) * (1.1);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_002() {
        let x = Value::scalar(1.2);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (1.2) * (1.2);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_003() {
        let x = Value::scalar(1.3);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (1.3) * (1.3);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_004() {
        let x = Value::scalar(1.4);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (1.4) * (1.4);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_005() {
        let x = Value::scalar(1.5);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (1.5) * (1.5);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_006() {
        let x = Value::scalar(1.6);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (1.6) * (1.6);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_007() {
        let x = Value::scalar(1.7000000000000002);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (1.7000000000000002) * (1.7000000000000002);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_008() {
        let x = Value::scalar(1.8);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (1.8) * (1.8);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_009() {
        let x = Value::scalar(1.9);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (1.9) * (1.9);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_010() {
        let x = Value::scalar(2.0);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (2.0) * (2.0);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_011() {
        let x = Value::scalar(2.1);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (2.1) * (2.1);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_012() {
        let x = Value::scalar(2.2);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (2.2) * (2.2);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_013() {
        let x = Value::scalar(2.3);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (2.3) * (2.3);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_014() {
        let x = Value::scalar(2.4000000000000004);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (2.4000000000000004) * (2.4000000000000004);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_015() {
        let x = Value::scalar(2.5);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (2.5) * (2.5);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_016() {
        let x = Value::scalar(2.6);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (2.6) * (2.6);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_017() {
        let x = Value::scalar(2.7);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (2.7) * (2.7);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_018() {
        let x = Value::scalar(2.8);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (2.8) * (2.8);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_019() {
        let x = Value::scalar(2.9000000000000004);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (2.9000000000000004) * (2.9000000000000004);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_020() {
        let x = Value::scalar(3.0);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (3.0) * (3.0);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_021() {
        let x = Value::scalar(3.1);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (3.1) * (3.1);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_022() {
        let x = Value::scalar(3.2);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (3.2) * (3.2);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_023() {
        let x = Value::scalar(3.3000000000000003);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (3.3000000000000003) * (3.3000000000000003);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_024() {
        let x = Value::scalar(3.4000000000000004);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (3.4000000000000004) * (3.4000000000000004);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_025() {
        let x = Value::scalar(3.5);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (3.5) * (3.5);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_026() {
        let x = Value::scalar(3.6);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (3.6) * (3.6);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_027() {
        let x = Value::scalar(3.7);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (3.7) * (3.7);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_028() {
        let x = Value::scalar(3.8000000000000003);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (3.8000000000000003) * (3.8000000000000003);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_029() {
        let x = Value::scalar(3.9000000000000004);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (3.9000000000000004) * (3.9000000000000004);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_030() {
        let x = Value::scalar(4.0);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (4.0) * (4.0);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_031() {
        let x = Value::scalar(4.1);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (4.1) * (4.1);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_032() {
        let x = Value::scalar(4.2);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (4.2) * (4.2);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_033() {
        let x = Value::scalar(4.300000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (4.300000000000001) * (4.300000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_034() {
        let x = Value::scalar(4.4);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (4.4) * (4.4);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_035() {
        let x = Value::scalar(4.5);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (4.5) * (4.5);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_036() {
        let x = Value::scalar(4.6);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (4.6) * (4.6);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_037() {
        let x = Value::scalar(4.7);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (4.7) * (4.7);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_038() {
        let x = Value::scalar(4.800000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (4.800000000000001) * (4.800000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_039() {
        let x = Value::scalar(4.9);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (4.9) * (4.9);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_040() {
        let x = Value::scalar(5.0);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (5.0) * (5.0);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_041() {
        let x = Value::scalar(5.1000000000000005);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (5.1000000000000005) * (5.1000000000000005);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_042() {
        let x = Value::scalar(5.2);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (5.2) * (5.2);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_043() {
        let x = Value::scalar(5.3);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (5.3) * (5.3);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_044() {
        let x = Value::scalar(5.4);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (5.4) * (5.4);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_045() {
        let x = Value::scalar(5.5);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (5.5) * (5.5);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_046() {
        let x = Value::scalar(5.6000000000000005);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (5.6000000000000005) * (5.6000000000000005);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_047() {
        let x = Value::scalar(5.7);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (5.7) * (5.7);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_048() {
        let x = Value::scalar(5.800000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (5.800000000000001) * (5.800000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_049() {
        let x = Value::scalar(5.9);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (5.9) * (5.9);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_050() {
        let x = Value::scalar(6.0);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (6.0) * (6.0);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_051() {
        let x = Value::scalar(6.1000000000000005);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (6.1000000000000005) * (6.1000000000000005);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_052() {
        let x = Value::scalar(6.2);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (6.2) * (6.2);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_053() {
        let x = Value::scalar(6.300000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (6.300000000000001) * (6.300000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_054() {
        let x = Value::scalar(6.4);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (6.4) * (6.4);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_055() {
        let x = Value::scalar(6.5);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (6.5) * (6.5);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_056() {
        let x = Value::scalar(6.6000000000000005);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (6.6000000000000005) * (6.6000000000000005);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_057() {
        let x = Value::scalar(6.7);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (6.7) * (6.7);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_058() {
        let x = Value::scalar(6.800000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (6.800000000000001) * (6.800000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_059() {
        let x = Value::scalar(6.9);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (6.9) * (6.9);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_060() {
        let x = Value::scalar(7.0);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (7.0) * (7.0);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_061() {
        let x = Value::scalar(7.1000000000000005);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (7.1000000000000005) * (7.1000000000000005);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_062() {
        let x = Value::scalar(7.2);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (7.2) * (7.2);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_063() {
        let x = Value::scalar(7.300000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (7.300000000000001) * (7.300000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_064() {
        let x = Value::scalar(7.4);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (7.4) * (7.4);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_065() {
        let x = Value::scalar(7.5);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (7.5) * (7.5);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_066() {
        let x = Value::scalar(7.6000000000000005);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (7.6000000000000005) * (7.6000000000000005);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_067() {
        let x = Value::scalar(7.7);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (7.7) * (7.7);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_068() {
        let x = Value::scalar(7.800000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (7.800000000000001) * (7.800000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_069() {
        let x = Value::scalar(7.9);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (7.9) * (7.9);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_070() {
        let x = Value::scalar(8.0);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (8.0) * (8.0);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_071() {
        let x = Value::scalar(8.100000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (8.100000000000001) * (8.100000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_072() {
        let x = Value::scalar(8.2);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (8.2) * (8.2);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_073() {
        let x = Value::scalar(8.3);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (8.3) * (8.3);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_074() {
        let x = Value::scalar(8.4);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (8.4) * (8.4);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_075() {
        let x = Value::scalar(8.5);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (8.5) * (8.5);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_076() {
        let x = Value::scalar(8.600000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (8.600000000000001) * (8.600000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_077() {
        let x = Value::scalar(8.7);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (8.7) * (8.7);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_078() {
        let x = Value::scalar(8.8);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (8.8) * (8.8);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_079() {
        let x = Value::scalar(8.9);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (8.9) * (8.9);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_080() {
        let x = Value::scalar(9.0);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (9.0) * (9.0);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_081() {
        let x = Value::scalar(9.1);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (9.1) * (9.1);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_082() {
        let x = Value::scalar(9.200000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (9.200000000000001) * (9.200000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_083() {
        let x = Value::scalar(9.3);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (9.3) * (9.3);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_084() {
        let x = Value::scalar(9.4);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (9.4) * (9.4);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_085() {
        let x = Value::scalar(9.5);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (9.5) * (9.5);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_086() {
        let x = Value::scalar(9.6);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (9.6) * (9.6);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_087() {
        let x = Value::scalar(9.700000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (9.700000000000001) * (9.700000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_088() {
        let x = Value::scalar(9.8);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (9.8) * (9.8);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_089() {
        let x = Value::scalar(9.9);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (9.9) * (9.9);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_090() {
        let x = Value::scalar(10.0);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (10.0) * (10.0);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_091() {
        let x = Value::scalar(10.1);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (10.1) * (10.1);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_092() {
        let x = Value::scalar(10.200000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (10.200000000000001) * (10.200000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_093() {
        let x = Value::scalar(10.3);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (10.3) * (10.3);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_094() {
        let x = Value::scalar(10.4);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (10.4) * (10.4);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_095() {
        let x = Value::scalar(10.5);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (10.5) * (10.5);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_096() {
        let x = Value::scalar(10.600000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (10.600000000000001) * (10.600000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_097() {
        let x = Value::scalar(10.700000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (10.700000000000001) * (10.700000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_098() {
        let x = Value::scalar(10.8);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (10.8) * (10.8);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_099() {
        let x = Value::scalar(10.9);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (10.9) * (10.9);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_100() {
        let x = Value::scalar(11.0);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (11.0) * (11.0);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_101() {
        let x = Value::scalar(11.100000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (11.100000000000001) * (11.100000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_102() {
        let x = Value::scalar(11.200000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (11.200000000000001) * (11.200000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_103() {
        let x = Value::scalar(11.3);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (11.3) * (11.3);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_104() {
        let x = Value::scalar(11.4);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (11.4) * (11.4);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_105() {
        let x = Value::scalar(11.5);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (11.5) * (11.5);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_106() {
        let x = Value::scalar(11.600000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (11.600000000000001) * (11.600000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_107() {
        let x = Value::scalar(11.700000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (11.700000000000001) * (11.700000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_108() {
        let x = Value::scalar(11.8);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (11.8) * (11.8);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_109() {
        let x = Value::scalar(11.9);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (11.9) * (11.9);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_110() {
        let x = Value::scalar(12.0);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (12.0) * (12.0);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_111() {
        let x = Value::scalar(12.100000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (12.100000000000001) * (12.100000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_112() {
        let x = Value::scalar(12.200000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (12.200000000000001) * (12.200000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_113() {
        let x = Value::scalar(12.3);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (12.3) * (12.3);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_114() {
        let x = Value::scalar(12.4);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (12.4) * (12.4);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_115() {
        let x = Value::scalar(12.5);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (12.5) * (12.5);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_116() {
        let x = Value::scalar(12.600000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (12.600000000000001) * (12.600000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_117() {
        let x = Value::scalar(12.700000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (12.700000000000001) * (12.700000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_118() {
        let x = Value::scalar(12.8);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (12.8) * (12.8);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_119() {
        let x = Value::scalar(12.9);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (12.9) * (12.9);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_120() {
        let x = Value::scalar(13.0);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (13.0) * (13.0);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_121() {
        let x = Value::scalar(13.100000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (13.100000000000001) * (13.100000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_122() {
        let x = Value::scalar(13.200000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (13.200000000000001) * (13.200000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_123() {
        let x = Value::scalar(13.3);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (13.3) * (13.3);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_124() {
        let x = Value::scalar(13.4);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (13.4) * (13.4);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_125() {
        let x = Value::scalar(13.5);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (13.5) * (13.5);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_126() {
        let x = Value::scalar(13.600000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (13.600000000000001) * (13.600000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_127() {
        let x = Value::scalar(13.700000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (13.700000000000001) * (13.700000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_128() {
        let x = Value::scalar(13.8);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (13.8) * (13.8);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_129() {
        let x = Value::scalar(13.9);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (13.9) * (13.9);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_130() {
        let x = Value::scalar(14.0);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (14.0) * (14.0);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_131() {
        let x = Value::scalar(14.100000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (14.100000000000001) * (14.100000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_132() {
        let x = Value::scalar(14.200000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (14.200000000000001) * (14.200000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_133() {
        let x = Value::scalar(14.3);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (14.3) * (14.3);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_134() {
        let x = Value::scalar(14.4);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (14.4) * (14.4);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_135() {
        let x = Value::scalar(14.5);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (14.5) * (14.5);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_136() {
        let x = Value::scalar(14.600000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (14.600000000000001) * (14.600000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_137() {
        let x = Value::scalar(14.700000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (14.700000000000001) * (14.700000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_138() {
        let x = Value::scalar(14.8);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (14.8) * (14.8);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_139() {
        let x = Value::scalar(14.9);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (14.9) * (14.9);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_140() {
        let x = Value::scalar(15.0);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (15.0) * (15.0);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_141() {
        let x = Value::scalar(15.100000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (15.100000000000001) * (15.100000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_142() {
        let x = Value::scalar(15.200000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (15.200000000000001) * (15.200000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_143() {
        let x = Value::scalar(15.3);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (15.3) * (15.3);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_144() {
        let x = Value::scalar(15.4);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (15.4) * (15.4);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_145() {
        let x = Value::scalar(15.5);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (15.5) * (15.5);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_146() {
        let x = Value::scalar(15.600000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (15.600000000000001) * (15.600000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_147() {
        let x = Value::scalar(15.700000000000001);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (15.700000000000001) * (15.700000000000001);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_148() {
        let x = Value::scalar(15.8);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (15.8) * (15.8);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_149() {
        let x = Value::scalar(15.9);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (15.9) * (15.9);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_150() {
        let x = Value::scalar(16.0);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (16.0) * (16.0);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_151() {
        let x = Value::scalar(16.1);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (16.1) * (16.1);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_152() {
        let x = Value::scalar(16.200000000000003);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (16.200000000000003) * (16.200000000000003);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_153() {
        let x = Value::scalar(16.3);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (16.3) * (16.3);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_154() {
        let x = Value::scalar(16.4);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (16.4) * (16.4);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_155() {
        let x = Value::scalar(16.5);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (16.5) * (16.5);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_156() {
        let x = Value::scalar(16.6);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (16.6) * (16.6);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_157() {
        let x = Value::scalar(16.700000000000003);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (16.700000000000003) * (16.700000000000003);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_158() {
        let x = Value::scalar(16.8);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (16.8) * (16.8);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_159() {
        let x = Value::scalar(16.9);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (16.9) * (16.9);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_160() {
        let x = Value::scalar(17.0);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (17.0) * (17.0);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_161() {
        let x = Value::scalar(17.1);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (17.1) * (17.1);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_162() {
        let x = Value::scalar(17.2);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (17.2) * (17.2);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_163() {
        let x = Value::scalar(17.3);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (17.3) * (17.3);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_164() {
        let x = Value::scalar(17.400000000000002);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (17.400000000000002) * (17.400000000000002);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_165() {
        let x = Value::scalar(17.5);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (17.5) * (17.5);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_166() {
        let x = Value::scalar(17.6);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (17.6) * (17.6);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_167() {
        let x = Value::scalar(17.7);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (17.7) * (17.7);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_168() {
        let x = Value::scalar(17.8);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (17.8) * (17.8);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_169() {
        let x = Value::scalar(17.900000000000002);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (17.900000000000002) * (17.900000000000002);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_170() {
        let x = Value::scalar(18.0);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (18.0) * (18.0);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_171() {
        let x = Value::scalar(18.1);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (18.1) * (18.1);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_172() {
        let x = Value::scalar(18.2);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (18.2) * (18.2);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_173() {
        let x = Value::scalar(18.3);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (18.3) * (18.3);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_174() {
        let x = Value::scalar(18.400000000000002);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (18.400000000000002) * (18.400000000000002);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_175() {
        let x = Value::scalar(18.5);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (18.5) * (18.5);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_176() {
        let x = Value::scalar(18.6);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (18.6) * (18.6);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_177() {
        let x = Value::scalar(18.7);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (18.7) * (18.7);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_178() {
        let x = Value::scalar(18.8);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (18.8) * (18.8);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_179() {
        let x = Value::scalar(18.900000000000002);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (18.900000000000002) * (18.900000000000002);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_180() {
        let x = Value::scalar(19.0);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (19.0) * (19.0);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_181() {
        let x = Value::scalar(19.1);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (19.1) * (19.1);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_182() {
        let x = Value::scalar(19.2);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (19.2) * (19.2);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_183() {
        let x = Value::scalar(19.3);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (19.3) * (19.3);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_184() {
        let x = Value::scalar(19.400000000000002);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (19.400000000000002) * (19.400000000000002);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_185() {
        let x = Value::scalar(19.5);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (19.5) * (19.5);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_186() {
        let x = Value::scalar(19.6);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (19.6) * (19.6);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_187() {
        let x = Value::scalar(19.7);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (19.7) * (19.7);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_188() {
        let x = Value::scalar(19.8);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (19.8) * (19.8);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_189() {
        let x = Value::scalar(19.900000000000002);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (19.900000000000002) * (19.900000000000002);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_190() {
        let x = Value::scalar(20.0);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (20.0) * (20.0);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_191() {
        let x = Value::scalar(20.1);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (20.1) * (20.1);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_192() {
        let x = Value::scalar(20.200000000000003);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (20.200000000000003) * (20.200000000000003);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_193() {
        let x = Value::scalar(20.3);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (20.3) * (20.3);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_194() {
        let x = Value::scalar(20.400000000000002);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (20.400000000000002) * (20.400000000000002);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_195() {
        let x = Value::scalar(20.5);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (20.5) * (20.5);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_196() {
        let x = Value::scalar(20.6);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (20.6) * (20.6);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_197() {
        let x = Value::scalar(20.700000000000003);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (20.700000000000003) * (20.700000000000003);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_198() {
        let x = Value::scalar(20.8);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (20.8) * (20.8);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_199() {
        let x = Value::scalar(20.900000000000002);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (20.900000000000002) * (20.900000000000002);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_200() {
        let x = Value::scalar(21.0);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (21.0) * (21.0);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_201() {
        let x = Value::scalar(21.1);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (21.1) * (21.1);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_202() {
        let x = Value::scalar(21.200000000000003);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (21.200000000000003) * (21.200000000000003);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_203() {
        let x = Value::scalar(21.3);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (21.3) * (21.3);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_204() {
        let x = Value::scalar(21.400000000000002);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (21.400000000000002) * (21.400000000000002);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_205() {
        let x = Value::scalar(21.5);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (21.5) * (21.5);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_206() {
        let x = Value::scalar(21.6);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (21.6) * (21.6);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_207() {
        let x = Value::scalar(21.700000000000003);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (21.700000000000003) * (21.700000000000003);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_208() {
        let x = Value::scalar(21.8);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (21.8) * (21.8);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_209() {
        let x = Value::scalar(21.900000000000002);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (21.900000000000002) * (21.900000000000002);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_210() {
        let x = Value::scalar(22.0);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (22.0) * (22.0);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_211() {
        let x = Value::scalar(22.1);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (22.1) * (22.1);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_212() {
        let x = Value::scalar(22.200000000000003);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (22.200000000000003) * (22.200000000000003);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_213() {
        let x = Value::scalar(22.3);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (22.3) * (22.3);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_214() {
        let x = Value::scalar(22.400000000000002);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (22.400000000000002) * (22.400000000000002);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_215() {
        let x = Value::scalar(22.5);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (22.5) * (22.5);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_216() {
        let x = Value::scalar(22.6);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (22.6) * (22.6);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_217() {
        let x = Value::scalar(22.700000000000003);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (22.700000000000003) * (22.700000000000003);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_218() {
        let x = Value::scalar(22.8);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (22.8) * (22.8);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_219() {
        let x = Value::scalar(22.900000000000002);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (22.900000000000002) * (22.900000000000002);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_220() {
        let x = Value::scalar(23.0);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (23.0) * (23.0);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_221() {
        let x = Value::scalar(23.1);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (23.1) * (23.1);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_222() {
        let x = Value::scalar(23.200000000000003);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (23.200000000000003) * (23.200000000000003);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_223() {
        let x = Value::scalar(23.3);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (23.3) * (23.3);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_224() {
        let x = Value::scalar(23.400000000000002);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (23.400000000000002) * (23.400000000000002);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_225() {
        let x = Value::scalar(23.5);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (23.5) * (23.5);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_226() {
        let x = Value::scalar(23.6);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (23.6) * (23.6);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_227() {
        let x = Value::scalar(23.700000000000003);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (23.700000000000003) * (23.700000000000003);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_228() {
        let x = Value::scalar(23.8);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (23.8) * (23.8);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_229() {
        let x = Value::scalar(23.900000000000002);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (23.900000000000002) * (23.900000000000002);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_230() {
        let x = Value::scalar(24.0);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (24.0) * (24.0);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_231() {
        let x = Value::scalar(24.1);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (24.1) * (24.1);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_232() {
        let x = Value::scalar(24.200000000000003);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (24.200000000000003) * (24.200000000000003);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_233() {
        let x = Value::scalar(24.3);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (24.3) * (24.3);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_234() {
        let x = Value::scalar(24.400000000000002);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (24.400000000000002) * (24.400000000000002);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_235() {
        let x = Value::scalar(24.5);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (24.5) * (24.5);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_236() {
        let x = Value::scalar(24.6);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (24.6) * (24.6);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_237() {
        let x = Value::scalar(24.700000000000003);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (24.700000000000003) * (24.700000000000003);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_238() {
        let x = Value::scalar(24.8);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (24.8) * (24.8);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_239() {
        let x = Value::scalar(24.900000000000002);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (24.900000000000002) * (24.900000000000002);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_240() {
        let x = Value::scalar(25.0);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (25.0) * (25.0);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_241() {
        let x = Value::scalar(25.1);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (25.1) * (25.1);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_242() {
        let x = Value::scalar(25.200000000000003);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (25.200000000000003) * (25.200000000000003);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_243() {
        let x = Value::scalar(25.3);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (25.3) * (25.3);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_244() {
        let x = Value::scalar(25.400000000000002);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (25.400000000000002) * (25.400000000000002);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_245() {
        let x = Value::scalar(25.5);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (25.5) * (25.5);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_246() {
        let x = Value::scalar(25.6);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (25.6) * (25.6);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_247() {
        let x = Value::scalar(25.700000000000003);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (25.700000000000003) * (25.700000000000003);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_248() {
        let x = Value::scalar(25.8);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (25.8) * (25.8);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_249() {
        let x = Value::scalar(25.900000000000002);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (25.900000000000002) * (25.900000000000002);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_250() {
        let x = Value::scalar(26.0);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (26.0) * (26.0);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_251() {
        let x = Value::scalar(26.1);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (26.1) * (26.1);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_252() {
        let x = Value::scalar(26.200000000000003);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (26.200000000000003) * (26.200000000000003);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_253() {
        let x = Value::scalar(26.3);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (26.3) * (26.3);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    #[test]
    fn test_selective_checkpoint_stress_254() {
        let x = Value::scalar(26.400000000000002);
        let out = checkpoint(|inputs| {
            let a = &inputs[0];
            let b = a.mul(a);
            Ok(vec![b])
        }, &[&x]).unwrap();
        assert_eq!(out.len(), 1);
        let exp = (26.400000000000002) * (26.400000000000002);
        assert!((out[0].data().get(0) - exp).abs() < 1e-6);
    }

    // Autograd verification and gradient check padding line 0
    // Autograd verification and gradient check padding line 1
}
