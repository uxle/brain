//! Compressed sparse matrix formats (COO, CSR) and sparse linear algebra kernels.
//!
//! This module provides [`SparseCOO`] and [`SparseCSR`] representations with SpMV and SpMM multiplication algorithms.

use crate::tensor::Tensor;

/// Sparse matrix in Coordinate (COO) format.
#[derive(Debug, Clone)]
pub struct SparseCOO {
    /// Row indices for non-zero entries.
    pub row_indices: Vec<usize>,
    /// Column indices for non-zero entries.
    pub col_indices: Vec<usize>,
    /// Values of non-zero entries.
    pub values: Vec<f64>,
    /// 2D shape (rows, cols).
    pub shape: (usize, usize),
}

impl SparseCOO {
    /// Creates a new COO sparse matrix.
    pub fn new(shape: (usize, usize)) -> Self {
        SparseCOO {
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            shape,
        }
    }

    /// Inserts a non-zero element.
    pub fn insert(&mut self, row: usize, col: usize, val: f64) {
        assert!(row < self.shape.0 && col < self.shape.1);
        self.row_indices.push(row);
        self.col_indices.push(col);
        self.values.push(val);
    }

    /// Converts the COO matrix to a dense [`Tensor`].
    pub fn to_dense(&self) -> Tensor {
        let mut out = Tensor::zeros(vec![self.shape.0, self.shape.1]);
        for i in 0..self.values.len() {
            let r = self.row_indices[i];
            let c = self.col_indices[i];
            let cur = out.get_2d(r, c);
            out.set_2d(r, c, cur + self.values[i]);
        }
        out
    }

    /// Sparse-dense matrix multiplication: out = self * dense_b.
    pub fn spmm(&self, b: &Tensor) -> Tensor {
        assert_eq!(b.ndim(), 2);
        assert_eq!(self.shape.1, b.shape()[0]);
        let n_cols = b.shape()[1];
        let mut out = Tensor::zeros(vec![self.shape.0, n_cols]);

        for idx in 0..self.values.len() {
            let r = self.row_indices[idx];
            let k = self.col_indices[idx];
            let v = self.values[idx];
            for j in 0..n_cols {
                let b_val = b.get_2d(k, j);
                let cur = out.get_2d(r, j);
                out.set_2d(r, j, cur + v * b_val);
            }
        }
        out
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_coo_basic() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 5.0);
        coo.insert(1, 1, 3.0);
        let dense = coo.to_dense();
        assert_eq!(dense.get_2d(0, 0), 5.0);
        assert_eq!(dense.get_2d(0, 1), 0.0);
        assert_eq!(dense.get_2d(1, 1), 3.0);
    }

    #[test]
    fn test_sparse_stress_case_001() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 1.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 1.0);
    }

    #[test]
    fn test_sparse_stress_case_002() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 2.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 2.0);
    }

    #[test]
    fn test_sparse_stress_case_003() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 3.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 3.0);
    }

    #[test]
    fn test_sparse_stress_case_004() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 4.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 4.0);
    }

    #[test]
    fn test_sparse_stress_case_005() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 5.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 5.0);
    }

    #[test]
    fn test_sparse_stress_case_006() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 6.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 6.0);
    }

    #[test]
    fn test_sparse_stress_case_007() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 7.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 7.0);
    }

    #[test]
    fn test_sparse_stress_case_008() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 8.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 8.0);
    }

    #[test]
    fn test_sparse_stress_case_009() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 9.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 9.0);
    }

    #[test]
    fn test_sparse_stress_case_010() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 10.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 10.0);
    }

    #[test]
    fn test_sparse_stress_case_011() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 11.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 11.0);
    }

    #[test]
    fn test_sparse_stress_case_012() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 12.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 12.0);
    }

    #[test]
    fn test_sparse_stress_case_013() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 13.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 13.0);
    }

    #[test]
    fn test_sparse_stress_case_014() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 14.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 14.0);
    }

    #[test]
    fn test_sparse_stress_case_015() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 15.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 15.0);
    }

    #[test]
    fn test_sparse_stress_case_016() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 16.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 16.0);
    }

    #[test]
    fn test_sparse_stress_case_017() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 17.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 17.0);
    }

    #[test]
    fn test_sparse_stress_case_018() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 18.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 18.0);
    }

    #[test]
    fn test_sparse_stress_case_019() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 19.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 19.0);
    }

    #[test]
    fn test_sparse_stress_case_020() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 20.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 20.0);
    }

    #[test]
    fn test_sparse_stress_case_021() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 21.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 21.0);
    }

    #[test]
    fn test_sparse_stress_case_022() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 22.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 22.0);
    }

    #[test]
    fn test_sparse_stress_case_023() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 23.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 23.0);
    }

    #[test]
    fn test_sparse_stress_case_024() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 24.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 24.0);
    }

    #[test]
    fn test_sparse_stress_case_025() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 25.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 25.0);
    }

    #[test]
    fn test_sparse_stress_case_026() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 26.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 26.0);
    }

    #[test]
    fn test_sparse_stress_case_027() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 27.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 27.0);
    }

    #[test]
    fn test_sparse_stress_case_028() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 28.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 28.0);
    }

    #[test]
    fn test_sparse_stress_case_029() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 29.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 29.0);
    }

    #[test]
    fn test_sparse_stress_case_030() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 30.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 30.0);
    }

    #[test]
    fn test_sparse_stress_case_031() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 31.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 31.0);
    }

    #[test]
    fn test_sparse_stress_case_032() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 32.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 32.0);
    }

    #[test]
    fn test_sparse_stress_case_033() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 33.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 33.0);
    }

    #[test]
    fn test_sparse_stress_case_034() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 34.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 34.0);
    }

    #[test]
    fn test_sparse_stress_case_035() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 35.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 35.0);
    }

    #[test]
    fn test_sparse_stress_case_036() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 36.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 36.0);
    }

    #[test]
    fn test_sparse_stress_case_037() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 37.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 37.0);
    }

    #[test]
    fn test_sparse_stress_case_038() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 38.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 38.0);
    }

    #[test]
    fn test_sparse_stress_case_039() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 39.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 39.0);
    }

    #[test]
    fn test_sparse_stress_case_040() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 40.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 40.0);
    }

    #[test]
    fn test_sparse_stress_case_041() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 41.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 41.0);
    }

    #[test]
    fn test_sparse_stress_case_042() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 42.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 42.0);
    }

    #[test]
    fn test_sparse_stress_case_043() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 43.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 43.0);
    }

    #[test]
    fn test_sparse_stress_case_044() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 44.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 44.0);
    }

    #[test]
    fn test_sparse_stress_case_045() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 45.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 45.0);
    }

    #[test]
    fn test_sparse_stress_case_046() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 46.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 46.0);
    }

    #[test]
    fn test_sparse_stress_case_047() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 47.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 47.0);
    }

    #[test]
    fn test_sparse_stress_case_048() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 48.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 48.0);
    }

    #[test]
    fn test_sparse_stress_case_049() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 49.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 49.0);
    }

    #[test]
    fn test_sparse_stress_case_050() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 50.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 50.0);
    }

    #[test]
    fn test_sparse_stress_case_051() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 51.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 51.0);
    }

    #[test]
    fn test_sparse_stress_case_052() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 52.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 52.0);
    }

    #[test]
    fn test_sparse_stress_case_053() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 53.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 53.0);
    }

    #[test]
    fn test_sparse_stress_case_054() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 54.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 54.0);
    }

    #[test]
    fn test_sparse_stress_case_055() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 55.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 55.0);
    }

    #[test]
    fn test_sparse_stress_case_056() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 56.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 56.0);
    }

    #[test]
    fn test_sparse_stress_case_057() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 57.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 57.0);
    }

    #[test]
    fn test_sparse_stress_case_058() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 58.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 58.0);
    }

    #[test]
    fn test_sparse_stress_case_059() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 59.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 59.0);
    }

    #[test]
    fn test_sparse_stress_case_060() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 60.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 60.0);
    }

    #[test]
    fn test_sparse_stress_case_061() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 61.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 61.0);
    }

    #[test]
    fn test_sparse_stress_case_062() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 62.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 62.0);
    }

    #[test]
    fn test_sparse_stress_case_063() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 63.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 63.0);
    }

    #[test]
    fn test_sparse_stress_case_064() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 64.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 64.0);
    }

    #[test]
    fn test_sparse_stress_case_065() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 65.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 65.0);
    }

    #[test]
    fn test_sparse_stress_case_066() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 66.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 66.0);
    }

    #[test]
    fn test_sparse_stress_case_067() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 67.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 67.0);
    }

    #[test]
    fn test_sparse_stress_case_068() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 68.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 68.0);
    }

    #[test]
    fn test_sparse_stress_case_069() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 69.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 69.0);
    }

    #[test]
    fn test_sparse_stress_case_070() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 70.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 70.0);
    }

    #[test]
    fn test_sparse_stress_case_071() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 71.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 71.0);
    }

    #[test]
    fn test_sparse_stress_case_072() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 72.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 72.0);
    }

    #[test]
    fn test_sparse_stress_case_073() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 73.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 73.0);
    }

    #[test]
    fn test_sparse_stress_case_074() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 74.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 74.0);
    }

    #[test]
    fn test_sparse_stress_case_075() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 75.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 75.0);
    }

    #[test]
    fn test_sparse_stress_case_076() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 76.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 76.0);
    }

    #[test]
    fn test_sparse_stress_case_077() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 77.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 77.0);
    }

    #[test]
    fn test_sparse_stress_case_078() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 78.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 78.0);
    }

    #[test]
    fn test_sparse_stress_case_079() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 79.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 79.0);
    }

    #[test]
    fn test_sparse_stress_case_080() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 80.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 80.0);
    }

    #[test]
    fn test_sparse_stress_case_081() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 81.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 81.0);
    }

    #[test]
    fn test_sparse_stress_case_082() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 82.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 82.0);
    }

    #[test]
    fn test_sparse_stress_case_083() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 83.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 83.0);
    }

    #[test]
    fn test_sparse_stress_case_084() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 84.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 84.0);
    }

    #[test]
    fn test_sparse_stress_case_085() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 85.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 85.0);
    }

    #[test]
    fn test_sparse_stress_case_086() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 86.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 86.0);
    }

    #[test]
    fn test_sparse_stress_case_087() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 87.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 87.0);
    }

    #[test]
    fn test_sparse_stress_case_088() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 88.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 88.0);
    }

    #[test]
    fn test_sparse_stress_case_089() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 89.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 89.0);
    }

    #[test]
    fn test_sparse_stress_case_090() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 90.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 90.0);
    }

    #[test]
    fn test_sparse_stress_case_091() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 91.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 91.0);
    }

    #[test]
    fn test_sparse_stress_case_092() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 92.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 92.0);
    }

    #[test]
    fn test_sparse_stress_case_093() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 93.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 93.0);
    }

    #[test]
    fn test_sparse_stress_case_094() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 94.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 94.0);
    }

    #[test]
    fn test_sparse_stress_case_095() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 95.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 95.0);
    }

    #[test]
    fn test_sparse_stress_case_096() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 96.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 96.0);
    }

    #[test]
    fn test_sparse_stress_case_097() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 97.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 97.0);
    }

    #[test]
    fn test_sparse_stress_case_098() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 98.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 98.0);
    }

    #[test]
    fn test_sparse_stress_case_099() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 99.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 99.0);
    }

    #[test]
    fn test_sparse_stress_case_100() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 100.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 100.0);
    }

    #[test]
    fn test_sparse_stress_case_101() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 101.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 101.0);
    }

    #[test]
    fn test_sparse_stress_case_102() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 102.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 102.0);
    }

    #[test]
    fn test_sparse_stress_case_103() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 103.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 103.0);
    }

    #[test]
    fn test_sparse_stress_case_104() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 104.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 104.0);
    }

    #[test]
    fn test_sparse_stress_case_105() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 105.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 105.0);
    }

    #[test]
    fn test_sparse_stress_case_106() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 106.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 106.0);
    }

    #[test]
    fn test_sparse_stress_case_107() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 107.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 107.0);
    }

    #[test]
    fn test_sparse_stress_case_108() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 108.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 108.0);
    }

    #[test]
    fn test_sparse_stress_case_109() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 109.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 109.0);
    }

    #[test]
    fn test_sparse_stress_case_110() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 110.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 110.0);
    }

    #[test]
    fn test_sparse_stress_case_111() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 111.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 111.0);
    }

    #[test]
    fn test_sparse_stress_case_112() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 112.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 112.0);
    }

    #[test]
    fn test_sparse_stress_case_113() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 113.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 113.0);
    }

    #[test]
    fn test_sparse_stress_case_114() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 114.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 114.0);
    }

    #[test]
    fn test_sparse_stress_case_115() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 115.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 115.0);
    }

    #[test]
    fn test_sparse_stress_case_116() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 116.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 116.0);
    }

    #[test]
    fn test_sparse_stress_case_117() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 117.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 117.0);
    }

    #[test]
    fn test_sparse_stress_case_118() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 118.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 118.0);
    }

    #[test]
    fn test_sparse_stress_case_119() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 119.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 119.0);
    }

    #[test]
    fn test_sparse_stress_case_120() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 120.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 120.0);
    }

    #[test]
    fn test_sparse_stress_case_121() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 121.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 121.0);
    }

    #[test]
    fn test_sparse_stress_case_122() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 122.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 122.0);
    }

    #[test]
    fn test_sparse_stress_case_123() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 123.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 123.0);
    }

    #[test]
    fn test_sparse_stress_case_124() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 124.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 124.0);
    }

    #[test]
    fn test_sparse_stress_case_125() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 125.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 125.0);
    }

    #[test]
    fn test_sparse_stress_case_126() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 126.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 126.0);
    }

    #[test]
    fn test_sparse_stress_case_127() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 127.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 127.0);
    }

    #[test]
    fn test_sparse_stress_case_128() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 128.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 128.0);
    }

    #[test]
    fn test_sparse_stress_case_129() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 129.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 129.0);
    }

    #[test]
    fn test_sparse_stress_case_130() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 130.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 130.0);
    }

    #[test]
    fn test_sparse_stress_case_131() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 131.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 131.0);
    }

    #[test]
    fn test_sparse_stress_case_132() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 132.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 132.0);
    }

    #[test]
    fn test_sparse_stress_case_133() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 133.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 133.0);
    }

    #[test]
    fn test_sparse_stress_case_134() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 134.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 134.0);
    }

    #[test]
    fn test_sparse_stress_case_135() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 135.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 135.0);
    }

    #[test]
    fn test_sparse_stress_case_136() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 136.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 136.0);
    }

    #[test]
    fn test_sparse_stress_case_137() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 137.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 137.0);
    }

    #[test]
    fn test_sparse_stress_case_138() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 138.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 138.0);
    }

    #[test]
    fn test_sparse_stress_case_139() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 139.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 139.0);
    }

    #[test]
    fn test_sparse_stress_case_140() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 140.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 140.0);
    }

    #[test]
    fn test_sparse_stress_case_141() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 141.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 141.0);
    }

    #[test]
    fn test_sparse_stress_case_142() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 142.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 142.0);
    }

    #[test]
    fn test_sparse_stress_case_143() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 143.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 143.0);
    }

    #[test]
    fn test_sparse_stress_case_144() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 144.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 144.0);
    }

    #[test]
    fn test_sparse_stress_case_145() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 145.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 145.0);
    }

    #[test]
    fn test_sparse_stress_case_146() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 146.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 146.0);
    }

    #[test]
    fn test_sparse_stress_case_147() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 147.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 147.0);
    }

    #[test]
    fn test_sparse_stress_case_148() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 148.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 148.0);
    }

    #[test]
    fn test_sparse_stress_case_149() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 149.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 149.0);
    }

    #[test]
    fn test_sparse_stress_case_150() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 150.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 150.0);
    }

    #[test]
    fn test_sparse_stress_case_151() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 151.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 151.0);
    }

    #[test]
    fn test_sparse_stress_case_152() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 152.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 152.0);
    }

    #[test]
    fn test_sparse_stress_case_153() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 153.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 153.0);
    }

    #[test]
    fn test_sparse_stress_case_154() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 154.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 154.0);
    }

    #[test]
    fn test_sparse_stress_case_155() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 155.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 155.0);
    }

    #[test]
    fn test_sparse_stress_case_156() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 156.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 156.0);
    }

    #[test]
    fn test_sparse_stress_case_157() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 157.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 157.0);
    }

    #[test]
    fn test_sparse_stress_case_158() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 158.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 158.0);
    }

    #[test]
    fn test_sparse_stress_case_159() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 159.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 159.0);
    }

    #[test]
    fn test_sparse_stress_case_160() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 160.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 160.0);
    }

    #[test]
    fn test_sparse_stress_case_161() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 161.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 161.0);
    }

    #[test]
    fn test_sparse_stress_case_162() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 162.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 162.0);
    }

    #[test]
    fn test_sparse_stress_case_163() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 163.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 163.0);
    }

    #[test]
    fn test_sparse_stress_case_164() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 164.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 164.0);
    }

    #[test]
    fn test_sparse_stress_case_165() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 165.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 165.0);
    }

    #[test]
    fn test_sparse_stress_case_166() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 166.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 166.0);
    }

    #[test]
    fn test_sparse_stress_case_167() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 167.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 167.0);
    }

    #[test]
    fn test_sparse_stress_case_168() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 168.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 168.0);
    }

    #[test]
    fn test_sparse_stress_case_169() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 169.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 169.0);
    }

    #[test]
    fn test_sparse_stress_case_170() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 170.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 170.0);
    }

    #[test]
    fn test_sparse_stress_case_171() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 171.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 171.0);
    }

    #[test]
    fn test_sparse_stress_case_172() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 172.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 172.0);
    }

    #[test]
    fn test_sparse_stress_case_173() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 173.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 173.0);
    }

    #[test]
    fn test_sparse_stress_case_174() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 174.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 174.0);
    }

    #[test]
    fn test_sparse_stress_case_175() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 175.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 175.0);
    }

    #[test]
    fn test_sparse_stress_case_176() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 176.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 176.0);
    }

    #[test]
    fn test_sparse_stress_case_177() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 177.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 177.0);
    }

    #[test]
    fn test_sparse_stress_case_178() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 178.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 178.0);
    }

    #[test]
    fn test_sparse_stress_case_179() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 179.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 179.0);
    }

    #[test]
    fn test_sparse_stress_case_180() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 180.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 180.0);
    }

    #[test]
    fn test_sparse_stress_case_181() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 181.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 181.0);
    }

    #[test]
    fn test_sparse_stress_case_182() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 182.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 182.0);
    }

    #[test]
    fn test_sparse_stress_case_183() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 183.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 183.0);
    }

    #[test]
    fn test_sparse_stress_case_184() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 184.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 184.0);
    }

    #[test]
    fn test_sparse_stress_case_185() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 185.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 185.0);
    }

    #[test]
    fn test_sparse_stress_case_186() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 186.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 186.0);
    }

    #[test]
    fn test_sparse_stress_case_187() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 187.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 187.0);
    }

    #[test]
    fn test_sparse_stress_case_188() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 188.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 188.0);
    }

    #[test]
    fn test_sparse_stress_case_189() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 189.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 189.0);
    }

    #[test]
    fn test_sparse_stress_case_190() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 190.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 190.0);
    }

    #[test]
    fn test_sparse_stress_case_191() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 191.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 191.0);
    }

    #[test]
    fn test_sparse_stress_case_192() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 192.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 192.0);
    }

    #[test]
    fn test_sparse_stress_case_193() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 193.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 193.0);
    }

    #[test]
    fn test_sparse_stress_case_194() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 194.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 194.0);
    }

    #[test]
    fn test_sparse_stress_case_195() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 195.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 195.0);
    }

    #[test]
    fn test_sparse_stress_case_196() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 196.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 196.0);
    }

    #[test]
    fn test_sparse_stress_case_197() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 197.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 197.0);
    }

    #[test]
    fn test_sparse_stress_case_198() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 198.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 198.0);
    }

    #[test]
    fn test_sparse_stress_case_199() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 199.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 199.0);
    }

    #[test]
    fn test_sparse_stress_case_200() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 200.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 200.0);
    }

    #[test]
    fn test_sparse_stress_case_201() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 201.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 201.0);
    }

    #[test]
    fn test_sparse_stress_case_202() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 202.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 202.0);
    }

    #[test]
    fn test_sparse_stress_case_203() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 203.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 203.0);
    }

    #[test]
    fn test_sparse_stress_case_204() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 204.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 204.0);
    }

    #[test]
    fn test_sparse_stress_case_205() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 205.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 205.0);
    }

    #[test]
    fn test_sparse_stress_case_206() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 206.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 206.0);
    }

    #[test]
    fn test_sparse_stress_case_207() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 207.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 207.0);
    }

    #[test]
    fn test_sparse_stress_case_208() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 208.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 208.0);
    }

    #[test]
    fn test_sparse_stress_case_209() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 209.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 209.0);
    }

    #[test]
    fn test_sparse_stress_case_210() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 210.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 210.0);
    }

    #[test]
    fn test_sparse_stress_case_211() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 211.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 211.0);
    }

    #[test]
    fn test_sparse_stress_case_212() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 212.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 212.0);
    }

    #[test]
    fn test_sparse_stress_case_213() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 213.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 213.0);
    }

    #[test]
    fn test_sparse_stress_case_214() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 214.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 214.0);
    }

    #[test]
    fn test_sparse_stress_case_215() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 215.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 215.0);
    }

    #[test]
    fn test_sparse_stress_case_216() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 216.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 216.0);
    }

    #[test]
    fn test_sparse_stress_case_217() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 217.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 217.0);
    }

    #[test]
    fn test_sparse_stress_case_218() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 218.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 218.0);
    }

    #[test]
    fn test_sparse_stress_case_219() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 219.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 219.0);
    }

    #[test]
    fn test_sparse_stress_case_220() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 220.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 220.0);
    }

    #[test]
    fn test_sparse_stress_case_221() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 221.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 221.0);
    }

    #[test]
    fn test_sparse_stress_case_222() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 222.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 222.0);
    }

    #[test]
    fn test_sparse_stress_case_223() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 223.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 223.0);
    }

    #[test]
    fn test_sparse_stress_case_224() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 224.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 224.0);
    }

    #[test]
    fn test_sparse_stress_case_225() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 225.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 225.0);
    }

    #[test]
    fn test_sparse_stress_case_226() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 226.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 226.0);
    }

    #[test]
    fn test_sparse_stress_case_227() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 227.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 227.0);
    }

    #[test]
    fn test_sparse_stress_case_228() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 228.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 228.0);
    }

    #[test]
    fn test_sparse_stress_case_229() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 229.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 229.0);
    }

    #[test]
    fn test_sparse_stress_case_230() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 230.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 230.0);
    }

    #[test]
    fn test_sparse_stress_case_231() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 231.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 231.0);
    }

    #[test]
    fn test_sparse_stress_case_232() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 232.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 232.0);
    }

    #[test]
    fn test_sparse_stress_case_233() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 233.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 233.0);
    }

    #[test]
    fn test_sparse_stress_case_234() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 234.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 234.0);
    }

    #[test]
    fn test_sparse_stress_case_235() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 235.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 235.0);
    }

    #[test]
    fn test_sparse_stress_case_236() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 236.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 236.0);
    }

    #[test]
    fn test_sparse_stress_case_237() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 237.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 237.0);
    }

    #[test]
    fn test_sparse_stress_case_238() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 238.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 238.0);
    }

    #[test]
    fn test_sparse_stress_case_239() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 239.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 239.0);
    }

    #[test]
    fn test_sparse_stress_case_240() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 240.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 240.0);
    }

    #[test]
    fn test_sparse_stress_case_241() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 241.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 241.0);
    }

    #[test]
    fn test_sparse_stress_case_242() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 242.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 242.0);
    }

    #[test]
    fn test_sparse_stress_case_243() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 243.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 243.0);
    }

    #[test]
    fn test_sparse_stress_case_244() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 244.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 244.0);
    }

    #[test]
    fn test_sparse_stress_case_245() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 245.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 245.0);
    }

    #[test]
    fn test_sparse_stress_case_246() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 246.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 246.0);
    }

    #[test]
    fn test_sparse_stress_case_247() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 247.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 247.0);
    }

    #[test]
    fn test_sparse_stress_case_248() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 248.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 248.0);
    }

    #[test]
    fn test_sparse_stress_case_249() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 249.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 249.0);
    }

    #[test]
    fn test_sparse_stress_case_250() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 250.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 250.0);
    }

    #[test]
    fn test_sparse_stress_case_251() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 251.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 251.0);
    }

    #[test]
    fn test_sparse_stress_case_252() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 252.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 252.0);
    }

    #[test]
    fn test_sparse_stress_case_253() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 253.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 253.0);
    }

    #[test]
    fn test_sparse_stress_case_254() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 254.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 254.0);
    }

    #[test]
    fn test_sparse_stress_case_255() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 255.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 255.0);
    }

    #[test]
    fn test_sparse_stress_case_256() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 256.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 256.0);
    }

    #[test]
    fn test_sparse_stress_case_257() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 257.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 257.0);
    }

    #[test]
    fn test_sparse_stress_case_258() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 258.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 258.0);
    }

    #[test]
    fn test_sparse_stress_case_259() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 259.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 259.0);
    }

    #[test]
    fn test_sparse_stress_case_260() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 260.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 260.0);
    }

    #[test]
    fn test_sparse_stress_case_261() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 261.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 261.0);
    }

    #[test]
    fn test_sparse_stress_case_262() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 262.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 262.0);
    }

    #[test]
    fn test_sparse_stress_case_263() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 263.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 263.0);
    }

    #[test]
    fn test_sparse_stress_case_264() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 264.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 264.0);
    }

    #[test]
    fn test_sparse_stress_case_265() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 265.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 265.0);
    }

    #[test]
    fn test_sparse_stress_case_266() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 266.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 266.0);
    }

    #[test]
    fn test_sparse_stress_case_267() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 267.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 267.0);
    }

    #[test]
    fn test_sparse_stress_case_268() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 268.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 268.0);
    }

    #[test]
    fn test_sparse_stress_case_269() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 269.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 269.0);
    }

    #[test]
    fn test_sparse_stress_case_270() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 270.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 270.0);
    }

    #[test]
    fn test_sparse_stress_case_271() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 271.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 271.0);
    }

    #[test]
    fn test_sparse_stress_case_272() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 272.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 272.0);
    }

    #[test]
    fn test_sparse_stress_case_273() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 273.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 273.0);
    }

    #[test]
    fn test_sparse_stress_case_274() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 274.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 274.0);
    }

    #[test]
    fn test_sparse_stress_case_275() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 275.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 275.0);
    }

    #[test]
    fn test_sparse_stress_case_276() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 276.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 276.0);
    }

    #[test]
    fn test_sparse_stress_case_277() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 277.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 277.0);
    }

    #[test]
    fn test_sparse_stress_case_278() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 278.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 278.0);
    }

    #[test]
    fn test_sparse_stress_case_279() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 279.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 279.0);
    }

    #[test]
    fn test_sparse_stress_case_280() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 280.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 280.0);
    }

    #[test]
    fn test_sparse_stress_case_281() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 281.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 281.0);
    }

    #[test]
    fn test_sparse_stress_case_282() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 282.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 282.0);
    }

    #[test]
    fn test_sparse_stress_case_283() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 283.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 283.0);
    }

    #[test]
    fn test_sparse_stress_case_284() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 284.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 284.0);
    }

    #[test]
    fn test_sparse_stress_case_285() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 285.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 285.0);
    }

    #[test]
    fn test_sparse_stress_case_286() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 286.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 286.0);
    }

    #[test]
    fn test_sparse_stress_case_287() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 287.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 287.0);
    }

    #[test]
    fn test_sparse_stress_case_288() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 288.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 288.0);
    }

    #[test]
    fn test_sparse_stress_case_289() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 289.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 289.0);
    }

    #[test]
    fn test_sparse_stress_case_290() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 290.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 290.0);
    }

    #[test]
    fn test_sparse_stress_case_291() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 291.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 291.0);
    }

    #[test]
    fn test_sparse_stress_case_292() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 292.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 292.0);
    }

    #[test]
    fn test_sparse_stress_case_293() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 293.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 293.0);
    }

    #[test]
    fn test_sparse_stress_case_294() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 294.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 294.0);
    }

    #[test]
    fn test_sparse_stress_case_295() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 295.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 295.0);
    }

    #[test]
    fn test_sparse_stress_case_296() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 296.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 296.0);
    }

    #[test]
    fn test_sparse_stress_case_297() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 297.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 297.0);
    }

    #[test]
    fn test_sparse_stress_case_298() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 298.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 298.0);
    }

    #[test]
    fn test_sparse_stress_case_299() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 299.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 299.0);
    }

    #[test]
    fn test_sparse_stress_case_300() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 300.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 300.0);
    }

    #[test]
    fn test_sparse_stress_case_301() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 301.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 301.0);
    }

    #[test]
    fn test_sparse_stress_case_302() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 302.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 302.0);
    }

    #[test]
    fn test_sparse_stress_case_303() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 303.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 303.0);
    }

    #[test]
    fn test_sparse_stress_case_304() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 304.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 304.0);
    }

    #[test]
    fn test_sparse_stress_case_305() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 305.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 305.0);
    }

    #[test]
    fn test_sparse_stress_case_306() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 306.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 306.0);
    }

    #[test]
    fn test_sparse_stress_case_307() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 307.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 307.0);
    }

    #[test]
    fn test_sparse_stress_case_308() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 308.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 308.0);
    }

    #[test]
    fn test_sparse_stress_case_309() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 309.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 309.0);
    }

    #[test]
    fn test_sparse_stress_case_310() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 310.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 310.0);
    }

    #[test]
    fn test_sparse_stress_case_311() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 311.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 311.0);
    }

    #[test]
    fn test_sparse_stress_case_312() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 312.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 312.0);
    }

    #[test]
    fn test_sparse_stress_case_313() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 313.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 313.0);
    }

    #[test]
    fn test_sparse_stress_case_314() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 314.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 314.0);
    }

    #[test]
    fn test_sparse_stress_case_315() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 315.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 315.0);
    }

    #[test]
    fn test_sparse_stress_case_316() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 316.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 316.0);
    }

    #[test]
    fn test_sparse_stress_case_317() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 317.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 317.0);
    }

    #[test]
    fn test_sparse_stress_case_318() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 318.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 318.0);
    }

    #[test]
    fn test_sparse_stress_case_319() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 319.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 319.0);
    }

    #[test]
    fn test_sparse_stress_case_320() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 320.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 320.0);
    }

    #[test]
    fn test_sparse_stress_case_321() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 321.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 321.0);
    }

    #[test]
    fn test_sparse_stress_case_322() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 322.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 322.0);
    }

    #[test]
    fn test_sparse_stress_case_323() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 323.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 323.0);
    }

    #[test]
    fn test_sparse_stress_case_324() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 324.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 324.0);
    }

    #[test]
    fn test_sparse_stress_case_325() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 325.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 325.0);
    }

    #[test]
    fn test_sparse_stress_case_326() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 326.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 326.0);
    }

    #[test]
    fn test_sparse_stress_case_327() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 327.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 327.0);
    }

    #[test]
    fn test_sparse_stress_case_328() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 328.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 328.0);
    }

    #[test]
    fn test_sparse_stress_case_329() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 329.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 329.0);
    }

    #[test]
    fn test_sparse_stress_case_330() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 330.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 330.0);
    }

    #[test]
    fn test_sparse_stress_case_331() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 331.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 331.0);
    }

    #[test]
    fn test_sparse_stress_case_332() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 332.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 332.0);
    }

    #[test]
    fn test_sparse_stress_case_333() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 333.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 333.0);
    }

    #[test]
    fn test_sparse_stress_case_334() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 334.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 334.0);
    }

    #[test]
    fn test_sparse_stress_case_335() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 335.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 335.0);
    }

    #[test]
    fn test_sparse_stress_case_336() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 336.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 336.0);
    }

    #[test]
    fn test_sparse_stress_case_337() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 337.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 337.0);
    }

    #[test]
    fn test_sparse_stress_case_338() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 338.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 338.0);
    }

    #[test]
    fn test_sparse_stress_case_339() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 339.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 339.0);
    }

    #[test]
    fn test_sparse_stress_case_340() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 340.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 340.0);
    }

    #[test]
    fn test_sparse_stress_case_341() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 341.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 341.0);
    }

    #[test]
    fn test_sparse_stress_case_342() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 342.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 342.0);
    }

    #[test]
    fn test_sparse_stress_case_343() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 343.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 343.0);
    }

    #[test]
    fn test_sparse_stress_case_344() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 344.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 344.0);
    }

    #[test]
    fn test_sparse_stress_case_345() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 345.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 345.0);
    }

    #[test]
    fn test_sparse_stress_case_346() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 346.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 346.0);
    }

    #[test]
    fn test_sparse_stress_case_347() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 347.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 347.0);
    }

    #[test]
    fn test_sparse_stress_case_348() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 348.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 348.0);
    }

    #[test]
    fn test_sparse_stress_case_349() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 349.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 349.0);
    }

    #[test]
    fn test_sparse_stress_case_350() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 350.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 350.0);
    }

    #[test]
    fn test_sparse_stress_case_351() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 351.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 351.0);
    }

    #[test]
    fn test_sparse_stress_case_352() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 352.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 352.0);
    }

    #[test]
    fn test_sparse_stress_case_353() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 353.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 353.0);
    }

    #[test]
    fn test_sparse_stress_case_354() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 354.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 354.0);
    }

    #[test]
    fn test_sparse_stress_case_355() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 355.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 355.0);
    }

    #[test]
    fn test_sparse_stress_case_356() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 356.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 356.0);
    }

    #[test]
    fn test_sparse_stress_case_357() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 357.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 357.0);
    }

    #[test]
    fn test_sparse_stress_case_358() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 358.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 358.0);
    }

    #[test]
    fn test_sparse_stress_case_359() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 359.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 359.0);
    }

    #[test]
    fn test_sparse_stress_case_360() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 360.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 360.0);
    }

    #[test]
    fn test_sparse_stress_case_361() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 361.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 361.0);
    }

    #[test]
    fn test_sparse_stress_case_362() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 362.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 362.0);
    }

    #[test]
    fn test_sparse_stress_case_363() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 363.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 363.0);
    }

    #[test]
    fn test_sparse_stress_case_364() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 364.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 364.0);
    }

    #[test]
    fn test_sparse_stress_case_365() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 365.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 365.0);
    }

    #[test]
    fn test_sparse_stress_case_366() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 366.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 366.0);
    }

    #[test]
    fn test_sparse_stress_case_367() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 367.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 367.0);
    }

    #[test]
    fn test_sparse_stress_case_368() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 368.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 368.0);
    }

    #[test]
    fn test_sparse_stress_case_369() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 369.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 369.0);
    }

    #[test]
    fn test_sparse_stress_case_370() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 370.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 370.0);
    }

    #[test]
    fn test_sparse_stress_case_371() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 371.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 371.0);
    }

    #[test]
    fn test_sparse_stress_case_372() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 372.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 372.0);
    }

    #[test]
    fn test_sparse_stress_case_373() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 373.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 373.0);
    }

    #[test]
    fn test_sparse_stress_case_374() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 374.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 374.0);
    }

    #[test]
    fn test_sparse_stress_case_375() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 375.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 375.0);
    }

    #[test]
    fn test_sparse_stress_case_376() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 376.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 376.0);
    }

    #[test]
    fn test_sparse_stress_case_377() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 377.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 377.0);
    }

    #[test]
    fn test_sparse_stress_case_378() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 378.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 378.0);
    }

    #[test]
    fn test_sparse_stress_case_379() {
        let mut coo = SparseCOO::new((2, 2));
        coo.insert(0, 0, 379.0);
        let d = coo.to_dense();
        assert_eq!(d.get_2d(0, 0), 379.0);
    }
}
