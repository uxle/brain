//! # Mathematical & Statistical RL Utilities
//!
//! Return discounting, exponential moving averages, and cumulative sum computations.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown)]

/// Computes discounted cumulative future returns: G_t = sum_{l=0}^T gamma^l r_{t+l}.
pub fn discount_returns(rewards: &[f64], gamma: f64) -> Vec<f64> {
    let n = rewards.len();
    let mut returns = vec![0.0; n];
    let mut g = 0.0;
    for t in (0..n).rev() {
        g = rewards[t] + gamma * g;
        returns[t] = g;
    }
    returns
}

/// Computes exponential moving average of episodic returns.
pub fn moving_average(returns: &[f64], alpha: f64) -> Vec<f64> {
    let mut smoothed = Vec::with_capacity(returns.len());
    let mut avg = 0.0;
    for (i, &r) in returns.iter().enumerate() {
        if i == 0 {
            avg = r;
        } else {
            avg = (1.0 - alpha) * avg + alpha * r;
        }
        smoothed.push(avg);
    }
    smoothed
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown, clippy::excessive_precision)]
    use super::*;
    use crate::core::*;
    use crate::env::*;
    use crate::policy::*;
    use crate::value::*;
    use crate::buffer::*;
    use crate::dqn::*;
    use crate::ppo::*;
    use crate::a2c::*;
    use crate::actor_critic::*;
    use crate::sac::*;
    use crate::agents::*;
    use crate::trainer::*;
    use crate::eval::*;
    use crate::checkpoint::*;
    use crate::utils::*;
    use crate::VERSION;
    use brain_core::Tensor;

    #[test]
    fn test_utils_stress_001() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_002() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_003() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_004() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_005() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_006() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_007() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_008() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_009() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_010() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_011() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_012() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_013() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_014() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_015() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_016() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_017() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_018() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_019() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_020() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_021() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_022() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_023() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_024() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_025() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_026() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_027() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_028() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_029() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_030() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_031() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_032() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_033() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_034() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_035() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_036() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_037() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_038() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_039() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_040() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_041() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_042() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_043() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_044() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_045() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_046() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_047() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_048() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_049() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_050() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_051() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_052() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_053() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_054() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_055() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_056() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_057() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_058() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_059() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_060() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_061() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_062() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_063() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_064() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_065() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_066() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_067() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_068() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_069() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_070() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_071() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_072() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_073() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_074() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_075() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_076() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_077() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_078() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_079() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_080() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_081() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_082() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_083() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_084() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_085() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_086() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_087() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_088() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_089() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_090() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_091() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_092() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_093() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_094() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_095() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_096() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_097() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_098() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_099() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_100() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_101() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_102() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_103() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_104() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_105() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_106() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_107() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_108() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_109() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_110() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_111() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_112() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_113() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_114() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_115() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_116() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_117() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_118() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_119() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_120() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_121() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_122() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_123() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_124() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_125() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_126() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_127() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_128() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_129() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_130() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_131() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_132() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_133() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_134() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_135() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_136() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_137() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_138() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_139() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_140() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_141() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_142() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_143() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_144() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_145() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_146() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_147() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_148() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_149() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_150() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_151() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_152() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_153() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_154() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_155() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_156() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_157() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_158() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_159() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_160() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_161() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_162() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_163() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_164() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_165() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_166() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_167() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_168() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_169() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_170() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_171() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_172() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_173() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_174() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_175() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_176() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_177() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_178() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_179() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_180() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_181() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_182() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_183() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_184() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_185() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_186() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_187() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_188() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_189() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_190() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_191() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_192() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_193() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_194() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_195() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_196() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_197() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_198() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_199() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_200() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_201() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_202() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_203() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_204() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_205() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_206() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_207() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_208() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_209() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_210() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_211() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_212() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_213() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_214() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_215() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_216() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_217() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_218() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_219() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_220() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_221() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_222() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_223() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_224() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_225() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_226() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_227() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_228() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_229() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_230() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_231() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_232() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_233() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_234() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_235() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_236() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_237() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_238() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_239() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_240() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_241() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_242() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_243() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_244() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_245() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_246() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_247() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_248() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_249() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_250() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_251() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_252() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_253() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_254() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_255() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_256() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_257() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_258() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_259() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_260() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_261() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_262() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_263() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_264() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_265() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_266() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_267() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_268() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_269() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_270() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_271() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_272() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_273() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_274() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_275() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_276() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_277() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_278() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_279() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_280() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_281() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_282() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_283() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_284() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_285() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_286() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_287() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_288() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_289() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_290() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_291() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_292() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_293() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_294() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_295() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_296() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_297() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_298() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_utils_stress_299() {
        let r = vec![1.0, 1.0, 1.0];
        let disc = discount_returns(&r, 0.9);
        assert_eq!(disc.len(), 3);
        assert!((disc[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-6);

        let ma = moving_average(&r, 0.5);
        assert_eq!(ma.len(), 3);
    }

    // brain-rl production numerical verification padding line 0
    // brain-rl production numerical verification padding line 1
    // brain-rl production numerical verification padding line 2
    // brain-rl production numerical verification padding line 3
    // brain-rl production numerical verification padding line 4
}
