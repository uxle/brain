# Phase 1: Audit & De-Duplicate Tests in `brain-core`

**Stage:** A — Test Audit & De-duplication
**Status:** ✅ Complete
**Pass Rate:** 32 / 32 numerical check tests passed

## Objective
Establish a clean, de-duplicated test suite for `brain-core` covering critical edge cases, non-contiguous strided views, IEEE 754 float propagation, SVD reconstruction, and FFT roundtrips.

## Key Verifications
1. **Empty Tensors & Scalars**:
   - `shape = [0]` numel is 0.
   - `shape = []` 0-dimensional scalar tensor correctly returns `.item()`.
2. **Non-Contiguous Strided Operations**:
   - Transposed $3 	imes 2$ matrix multiplication with $2 	imes 2$ identity produces correct result without copy errors.
   - Reductions over transposed views compute accurate sums.
3. **NaN & Inf Propagation**:
   - `[1.0, NaN, 3.0]` + `[1.0, 2.0, 3.0]` preserves NaN at index 1 without panicking.
   - `[Inf, 1.0]` * 2.0 produces Inf at index 0.
4. **Determinant Reference**:
   - $4 	imes 4$ diagonal matrix determinant verified ($120.0$).
   - $8 	imes 8$ scaled matrix determinant verified ($2^8 = 256.0$).
5. **SVD Reconstruction Fidelity**:
   - SVD decomposition $A = U \Sigma V^T$ reconstructed with Frobenius error $\|A - \hat{A}\|_F < 10^{-7}$.
6. **FFT / IFFT Roundtrip**:
   - Power-of-two length ($N=64$) and non-power-of-two arbitrary length ($N=50$) verified with max absolute error $< 10^{-6}$.
