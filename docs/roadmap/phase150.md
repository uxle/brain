# Phase 150: Adam & AdamW (Decoupled Weight Decay)

**Stage:** F — Loss Functions, Optimizers & Training Engine
**Status:** ✅ Complete
**Pass Rate:** Verified in `optim_step_test.rs`

## Objective
Verify first and second moment tracking ($m_t, v_t$), bias correction, and decoupled weight decay $\theta \leftarrow \theta (1 - \eta \lambda)$.
