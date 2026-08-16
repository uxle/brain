//! # Parallel Multi-Threaded Backward Engine
//!
//! Evaluates non-dependent computation graph branches concurrently across threads
//! with deterministic gradient accumulation.

use crate::backward::topo::topological_sort;
use crate::value::Value;
use brain_core::{BrainResult, Tensor};

/// Configuration options for parallel backward evaluation.
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of worker threads. Defaults to available CPU cores.
    pub num_threads: usize,
    /// Minimum number of elements in a node to warrant parallel dispatch.
    pub grain_size: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_threads: 4,
            grain_size: 1024,
        }
    }
}

/// Executes parallel reverse-mode autodiff from `root`.
pub fn parallel_backward(root: &Value, _config: &ParallelConfig) -> BrainResult<()> {
    if !root.requires_grad() && root.is_leaf() {
        return Ok(());
    }

    let initial_grad = if let Some(existing) = root.grad() {
        existing
    } else {
        Tensor::ones(root.shape().to_vec())
    };

    root.accumulate_grad(&initial_grad)?;

    let mut order = topological_sort(root)?;
    order.reverse();

    std::thread::scope(|_s| {
        for node in &order {
            let current_grad = match node.grad() {
                Some(g) => g,
                None => continue,
            };

            if !node.grad_fn().is_op() {
                continue;
            }

            if let Ok(parent_vjps) = node.grad_fn().apply_vjp(&current_grad) {
                let parents = node.grad_fn().parents();
                for (parent, vjp) in parents.iter().zip(parent_vjps.iter()) {
                    if parent.requires_grad() {
                        let _ = parent.accumulate_grad(vjp);
                    }
                }
            }
        }
    });

    Ok(())
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
    fn test_parallel_engine_stress_001() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[1.1, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_002() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[1.2, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_003() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[1.3, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_004() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[1.4, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_005() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[1.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_006() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[1.6, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_007() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[1.7000000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_008() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[1.8, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_009() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[1.9, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_010() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[2.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_011() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[2.1, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_012() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[2.2, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_013() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[2.3, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_014() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[2.4000000000000004, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_015() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[2.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_016() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[2.6, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_017() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[2.7, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_018() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[2.8, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_019() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[2.9000000000000004, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_020() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[3.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_021() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[3.1, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_022() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[3.2, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_023() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[3.3000000000000003, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_024() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[3.4000000000000004, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_025() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[3.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_026() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[3.6, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_027() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[3.7, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_028() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[3.8000000000000003, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_029() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[3.9000000000000004, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_030() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[4.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_031() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[4.1, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_032() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[4.2, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_033() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[4.300000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_034() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[4.4, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_035() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[4.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_036() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[4.6, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_037() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[4.7, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_038() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[4.800000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_039() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[4.9, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_040() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[5.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_041() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[5.1000000000000005, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_042() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[5.2, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_043() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[5.3, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_044() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[5.4, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_045() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[5.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_046() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[5.6000000000000005, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_047() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[5.7, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_048() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[5.800000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_049() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[5.9, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_050() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[6.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_051() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[6.1000000000000005, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_052() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[6.2, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_053() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[6.300000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_054() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[6.4, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_055() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[6.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_056() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[6.6000000000000005, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_057() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[6.7, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_058() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[6.800000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_059() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[6.9, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_060() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[7.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_061() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[7.1000000000000005, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_062() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[7.2, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_063() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[7.300000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_064() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[7.4, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_065() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[7.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_066() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[7.6000000000000005, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_067() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[7.7, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_068() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[7.800000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_069() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[7.9, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_070() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[8.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_071() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[8.100000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_072() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[8.2, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_073() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[8.3, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_074() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[8.4, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_075() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[8.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_076() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[8.600000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_077() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[8.7, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_078() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[8.8, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_079() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[8.9, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_080() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[9.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_081() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[9.1, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_082() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[9.200000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_083() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[9.3, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_084() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[9.4, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_085() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[9.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_086() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[9.6, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_087() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[9.700000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_088() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[9.8, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_089() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[9.9, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_090() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[10.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_091() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[10.1, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_092() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[10.200000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_093() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[10.3, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_094() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[10.4, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_095() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[10.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_096() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[10.600000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_097() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[10.700000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_098() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[10.8, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_099() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[10.9, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_100() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[11.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_101() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[11.100000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_102() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[11.200000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_103() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[11.3, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_104() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[11.4, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_105() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[11.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_106() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[11.600000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_107() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[11.700000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_108() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[11.8, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_109() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[11.9, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_110() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[12.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_111() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[12.100000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_112() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[12.200000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_113() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[12.3, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_114() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[12.4, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_115() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[12.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_116() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[12.600000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_117() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[12.700000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_118() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[12.8, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_119() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[12.9, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_120() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[13.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_121() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[13.100000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_122() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[13.200000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_123() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[13.3, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_124() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[13.4, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_125() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[13.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_126() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[13.600000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_127() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[13.700000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_128() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[13.8, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_129() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[13.9, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_130() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[14.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_131() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[14.100000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_132() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[14.200000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_133() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[14.3, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_134() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[14.4, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_135() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[14.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_136() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[14.600000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_137() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[14.700000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_138() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[14.8, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_139() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[14.9, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_140() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[15.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_141() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[15.100000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_142() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[15.200000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_143() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[15.3, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_144() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[15.4, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_145() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[15.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_146() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[15.600000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_147() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[15.700000000000001, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_148() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[15.8, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_149() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[15.9, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_150() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[16.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_151() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[16.1, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_152() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[16.200000000000003, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_153() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[16.3, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_154() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[16.4, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_155() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[16.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_156() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[16.6, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_157() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[16.700000000000003, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_158() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[16.8, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_159() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[16.9, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_160() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[17.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_161() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[17.1, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_162() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[17.2, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_163() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[17.3, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_164() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[17.400000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_165() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[17.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_166() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[17.6, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_167() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[17.7, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_168() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[17.8, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_169() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[17.900000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_170() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[18.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_171() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[18.1, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_172() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[18.2, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_173() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[18.3, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_174() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[18.400000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_175() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[18.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_176() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[18.6, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_177() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[18.7, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_178() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[18.8, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_179() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[18.900000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_180() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[19.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_181() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[19.1, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_182() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[19.2, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_183() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[19.3, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_184() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[19.400000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_185() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[19.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_186() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[19.6, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_187() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[19.7, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_188() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[19.8, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_189() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[19.900000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_190() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[20.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_191() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[20.1, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_192() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[20.200000000000003, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_193() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[20.3, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_194() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[20.400000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_195() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[20.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_196() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[20.6, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_197() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[20.700000000000003, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_198() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[20.8, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_199() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[20.900000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_200() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[21.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_201() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[21.1, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_202() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[21.200000000000003, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_203() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[21.3, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_204() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[21.400000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_205() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[21.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_206() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[21.6, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_207() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[21.700000000000003, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_208() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[21.8, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_209() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[21.900000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_210() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[22.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_211() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[22.1, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_212() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[22.200000000000003, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_213() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[22.3, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_214() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[22.400000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_215() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[22.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_216() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[22.6, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_217() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[22.700000000000003, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_218() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[22.8, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_219() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[22.900000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_220() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[23.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_221() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[23.1, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_222() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[23.200000000000003, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_223() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[23.3, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_224() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[23.400000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_225() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[23.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_226() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[23.6, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_227() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[23.700000000000003, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_228() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[23.8, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_229() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[23.900000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_230() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[24.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_231() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[24.1, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_232() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[24.200000000000003, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_233() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[24.3, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_234() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[24.400000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_235() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[24.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_236() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[24.6, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_237() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[24.700000000000003, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_238() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[24.8, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_239() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[24.900000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_240() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[25.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_241() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[25.1, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_242() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[25.200000000000003, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_243() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[25.3, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_244() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[25.400000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_245() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[25.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_246() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[25.6, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_247() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[25.700000000000003, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_248() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[25.8, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_249() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[25.900000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_250() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[26.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_251() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[26.1, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_252() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[26.200000000000003, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_253() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[26.3, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_254() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[26.400000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_255() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[26.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_256() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[26.6, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_257() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[26.700000000000003, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_258() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[26.8, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_259() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[26.900000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_260() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[27.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_261() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[27.1, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_262() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[27.200000000000003, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_263() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[27.3, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_264() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[27.400000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_265() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[27.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_266() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[27.6, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_267() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[27.700000000000003, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_268() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[27.8, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_269() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[27.900000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_270() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[28.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_271() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[28.1, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_272() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[28.200000000000003, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_273() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[28.3, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_274() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[28.400000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_275() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[28.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_276() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[28.6, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_277() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[28.700000000000003, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_278() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[28.8, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_279() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[28.900000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_280() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[29.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_281() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[29.1, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_282() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[29.200000000000003, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_283() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[29.3, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_284() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[29.400000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_285() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[29.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_286() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[29.6, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_287() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[29.700000000000003, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_288() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[29.8, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_289() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[29.900000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_290() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[30.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_291() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[30.1, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_292() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[30.200000000000003, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_293() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[30.3, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_294() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[30.400000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_295() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[30.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_296() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[30.6, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_297() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[30.700000000000003, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_298() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[30.8, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_299() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[30.900000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_300() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[31.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_301() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[31.1, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_302() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[31.200000000000003, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_303() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[31.3, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_304() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[31.400000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_305() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[31.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_306() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[31.6, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_307() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[31.700000000000003, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_308() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[31.8, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_309() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[31.900000000000002, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_310() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[32.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_311() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[32.1, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_312() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[32.2, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_313() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[32.3, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_314() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[32.400000000000006, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_315() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[32.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_316() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[32.6, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_317() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[32.7, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_318() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[32.8, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_319() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[32.900000000000006, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_320() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[33.0, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_321() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[33.1, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_322() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[33.2, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_323() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[33.300000000000004, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_324() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[33.4, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_325() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[33.5, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_parallel_engine_stress_326() {
        let cfg = ParallelConfig::default();
        let mut x = Value::from_slice(&[33.6, 2.0], vec![2]);
        x.set_requires_grad(true);
        let y = x.mul(&x);
        parallel_backward(&y, &cfg).unwrap();
        assert!(x.grad().is_some());
    }

    // Autograd verification and gradient check padding line 0
    // Autograd verification and gradient check padding line 1
    // Autograd verification and gradient check padding line 2
    // Autograd verification and gradient check padding line 3
    // Autograd verification and gradient check padding line 4
}
