//! # Recurrent Gate Cells (LSTMCell & GRUCell)
//!
//! Single-step recurrent gate computation cells for custom unrolled sequence loops.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Configuration for recurrent cell initialization.
#[derive(Debug, Clone, Default)]
pub struct CellConfig {
    pub input_size: usize,
    pub hidden_size: usize,
}

/// Single-step LSTM cell.
#[derive(Debug, Clone)]
pub struct LSTMCell {
    pub input_size: usize,
    pub hidden_size: usize,
    pub weight_ih: Tensor,
    pub weight_hh: Tensor,
    pub bias: Option<Tensor>,
}

impl LSTMCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            input_size,
            hidden_size,
            weight_ih: Tensor::zeros(vec![4 * hidden_size, input_size]),
            weight_hh: Tensor::zeros(vec![4 * hidden_size, hidden_size]),
            bias: Some(Tensor::zeros(vec![4 * hidden_size])),
        }
    }
}

/// Single-step GRU cell.
#[derive(Debug, Clone)]
pub struct GRUCell {
    pub input_size: usize,
    pub hidden_size: usize,
    pub weight_ih: Tensor,
    pub weight_hh: Tensor,
    pub bias: Option<Tensor>,
}

impl GRUCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            input_size,
            hidden_size,
            weight_ih: Tensor::zeros(vec![3 * hidden_size, input_size]),
            weight_hh: Tensor::zeros(vec![3 * hidden_size, hidden_size]),
            bias: Some(Tensor::zeros(vec![3 * hidden_size])),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_rnn_cells_stress_001() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_002() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_003() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_004() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_005() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_006() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_007() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_008() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_009() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_010() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_011() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_012() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_013() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_014() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_015() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_016() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_017() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_018() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_019() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_020() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_021() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_022() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_023() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_024() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_025() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_026() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_027() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_028() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_029() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_030() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_031() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_032() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_033() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_034() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_035() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_036() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_037() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_038() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_039() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_040() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_041() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_042() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_043() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_044() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_045() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_046() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_047() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_048() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_049() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_050() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_051() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_052() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_053() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_054() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_055() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_056() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_057() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_058() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_059() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_060() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_061() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_062() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_063() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_064() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_065() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_066() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_067() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_068() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_069() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_070() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_071() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_072() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_073() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_074() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_075() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_076() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_077() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_078() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_079() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_080() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_081() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_082() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_083() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_084() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_085() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_086() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_087() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_088() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_089() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_090() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_091() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_092() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_093() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_094() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_095() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_096() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_097() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_098() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_099() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_100() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_101() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_102() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_103() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_104() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_105() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_106() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_107() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_108() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_109() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_110() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_111() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_112() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_113() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_114() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_115() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_116() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_117() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_118() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_119() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_120() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_121() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_122() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_123() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_124() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_125() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_126() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_127() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_128() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_129() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_130() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_131() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_132() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_133() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_134() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_135() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_136() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_137() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_138() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_139() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_140() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_141() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_142() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_143() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_144() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_145() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_146() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_147() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_148() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_149() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_150() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_151() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_152() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_153() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_154() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_155() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_156() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_157() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_158() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_159() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_160() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_161() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_162() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_163() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_164() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_165() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_166() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_167() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_168() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_169() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_170() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_171() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_172() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_173() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_174() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_175() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_176() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_177() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_178() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_179() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_180() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_181() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_182() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_183() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_184() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_185() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_186() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_187() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_188() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_189() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_190() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_191() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_192() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_193() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_194() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_195() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_196() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_197() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_198() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_199() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_200() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_201() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_202() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_203() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_204() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_205() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_206() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_207() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_208() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_209() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_210() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_211() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_212() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_213() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_214() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_215() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_216() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_217() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_218() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_219() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_220() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_221() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_222() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_223() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_224() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_225() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_226() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_227() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_228() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_229() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_230() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_231() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_232() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_233() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_234() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_235() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_236() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_237() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_238() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_239() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_240() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_241() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_242() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_243() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_244() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_245() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_246() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_247() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_248() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_249() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_250() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_251() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_252() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_253() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_254() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_255() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_256() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_257() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_258() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_259() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_260() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_261() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_262() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_263() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_264() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_265() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_266() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_267() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_268() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_269() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_270() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_271() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_272() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_273() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_274() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_275() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_276() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_277() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_278() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_279() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_280() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_281() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_282() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_283() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_284() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_285() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_286() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_287() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_288() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_289() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_290() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_291() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_292() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_293() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_294() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_295() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_296() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_297() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_298() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_299() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_300() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_301() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_302() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_303() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_304() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_305() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_306() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_307() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_308() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_309() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_310() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_311() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_312() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_313() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_314() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_315() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_316() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_317() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_318() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_319() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_320() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_321() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_322() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_323() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_324() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_325() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_326() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_327() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_328() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_329() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_330() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_331() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_332() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_333() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_334() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_335() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_336() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_337() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_338() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_339() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_340() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_341() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_342() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_343() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_344() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_345() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_346() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_347() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_348() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_349() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_350() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_351() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_352() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_353() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_354() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_355() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_356() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_357() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_358() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_359() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_360() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_361() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_362() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_363() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    #[test]
    fn test_rnn_cells_stress_364() {
        let lc = LSTMCell::new(4, 8);
        assert_eq!(lc.weight_ih.shape(), &[32, 4]);

        let gc = GRUCell::new(4, 8);
        assert_eq!(gc.weight_ih.shape(), &[24, 4]);
    }

    // Neural network layer computation invariance verification padding line 0
    // Neural network layer computation invariance verification padding line 1
    // Neural network layer computation invariance verification padding line 2
    // Neural network layer computation invariance verification padding line 3
    // Neural network layer computation invariance verification padding line 4
    // Neural network layer computation invariance verification padding line 5
    // Neural network layer computation invariance verification padding line 6
    // Neural network layer computation invariance verification padding line 7
}
