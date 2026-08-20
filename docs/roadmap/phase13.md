# Phase 13: Audit & De-Duplicate Tests in `brain-metric`

**Stage:** A — Test Audit & De-duplication
**Status:** ✅ Complete
**Pass Rate:** 2 / 2 tests passed

## Objective
Verify ranking and classification evaluation metrics under ideal and inverted prediction distributions.

## Key Verifications
1. **ROC-AUC & PR-AUC**: Invariant boundary checks (1.0 on perfect ranking, 0.0 on inverted).
2. **Calibration**: Reliability diagrams and expected calibration errors.
