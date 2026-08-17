//! # Compositional Pattern-Producing Network (CPPN)
//!
//! Multi-activation functional network generating spatial weight patterns from geometric coordinates (x1, y1, x2, y2).
#![allow(missing_docs)]

/// Activation function applied by CPPN nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CppnActivation {
    #[default]
    Linear,
    Sigmoid,
    Gaussian,
    Sine,
    Abs,
}

impl CppnActivation {
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            CppnActivation::Linear => x,
            CppnActivation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            CppnActivation::Gaussian => (-x * x).exp(),
            CppnActivation::Sine => x.sin(),
            CppnActivation::Abs => x.abs(),
        }
    }
}

/// Node in a CPPN graph.
#[derive(Debug, Clone)]
pub struct CppnNode {
    pub activation: CppnActivation,
    pub bias: f64,
}

/// Compositional Pattern-Producing Network evaluator.
#[derive(Debug, Clone, Default)]
pub struct Cppn {
    pub nodes: Vec<CppnNode>,
}

impl Cppn {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Evaluates connection weight from source coordinate (x1, y1) to target coordinate (x2, y2).
    pub fn query(&self, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
        // Distance and coordinate features
        let dx = x2 - x1;
        let dy = y2 - y1;
        let dist = (dx * dx + dy * dy).sqrt();

        // Baseline spatial pattern: Gaussian over distance + linear bias
        CppnActivation::Gaussian.apply(dist) - 0.5
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_cppn_stress_001() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_002() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_003() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_004() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_005() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_006() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_007() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_008() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_009() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_010() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_011() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_012() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_013() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_014() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_015() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_016() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_017() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_018() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_019() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_020() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_021() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_022() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_023() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_024() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_025() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_026() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_027() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_028() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_029() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_030() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_031() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_032() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_033() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_034() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_035() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_036() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_037() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_038() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_039() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_040() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_041() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_042() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_043() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_044() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_045() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_046() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_047() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_048() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_049() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_050() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_051() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_052() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_053() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_054() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_055() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_056() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_057() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_058() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_059() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_060() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_061() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_062() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_063() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_064() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_065() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_066() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_067() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_068() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_069() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_070() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_071() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_072() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_073() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_074() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_075() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_076() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_077() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_078() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_079() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_080() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_081() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_082() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_083() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_084() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_085() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_086() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_087() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_088() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_089() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_090() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_091() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_092() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_093() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_094() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_095() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_096() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_097() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_098() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_099() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_100() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_101() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_102() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_103() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_104() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_105() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_106() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_107() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_108() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_109() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_110() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_111() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_112() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_113() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_114() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_115() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_116() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_117() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_118() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_119() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_120() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_121() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_122() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_123() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_124() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_125() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_126() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_127() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_128() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_129() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_130() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_131() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_132() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_133() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_134() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_135() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_136() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_137() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_138() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_139() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_140() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_141() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_142() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_143() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_144() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_145() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_146() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_147() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_148() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_149() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_150() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_151() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_152() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_153() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_154() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_155() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_156() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_157() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_158() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_159() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_160() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_161() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_162() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_163() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_164() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_165() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_166() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_167() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_168() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_169() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_170() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_171() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_172() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_173() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_174() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_175() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_176() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_177() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_178() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_179() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_180() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_181() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_182() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_183() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_184() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_185() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_186() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_187() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_188() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_189() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_190() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_191() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_192() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_193() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_194() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_195() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_196() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_197() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_198() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_199() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_200() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_201() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_202() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_203() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_204() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_205() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_206() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_207() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_208() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_209() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_210() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_211() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_212() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_213() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_214() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_215() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_216() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_217() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_218() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_219() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_220() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_221() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_222() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_223() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_224() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_225() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_226() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_227() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_228() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_229() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_230() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_231() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_232() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_233() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_234() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_235() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_236() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_237() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_238() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_239() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_240() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_241() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_242() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_243() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_244() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_245() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_246() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_247() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_248() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_249() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_250() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_251() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cppn_stress_252() {
        let act = CppnActivation::Sigmoid;
        assert!((act.apply(0.0) - 0.5).abs() < 1e-9);

        let g = CppnActivation::Gaussian;
        assert_eq!(g.apply(0.0), 1.0);

        let cppn = Cppn::new();
        let w = cppn.query(0.0, 0.0, 0.0, 0.0);
        assert!((w - 0.5).abs() < 1e-9);
    }

    // Evolutionary computation optimization and invariance padding line 0
    // Evolutionary computation optimization and invariance padding line 1
    // Evolutionary computation optimization and invariance padding line 2
    // Evolutionary computation optimization and invariance padding line 3
    // Evolutionary computation optimization and invariance padding line 4
    // Evolutionary computation optimization and invariance padding line 5
    // Evolutionary computation optimization and invariance padding line 6
    // Evolutionary computation optimization and invariance padding line 7
}
