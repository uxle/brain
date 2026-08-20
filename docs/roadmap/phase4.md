# Phase 4: Audit & De-Duplicate Tests in `brain-optim` + Optimizer Correctness

**Stage:** A — Test Audit & De-duplication
**Status:** ✅ Complete
**Pass Rate:** 13 / 13 trajectory tests passed

## Objective
Validate closed-form multi-step trajectory accuracy to $10^{-10}$ precision for SGD, Nesterov, Adam, AdamW (decoupled decay), RMSprop, Adagrad, and Lion.

## Key Verifications
1. **SGD with Momentum**: 5-step exact hand-calculated trajectory.
2. **SGD with Nesterov**: $\theta_t = \theta_{t-1} - \eta(g_t + \mu v_t)$ look-ahead trajectory.
3. **Adam & AdamW**: First/second moment updates, bias corrections, and decoupled weight decay $\theta_t = (1 - \eta \lambda)\theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$.
4. **Global Norm Clipping**: $g_i \leftarrow g_i \cdot \min\left(1, \frac{\text{clip}}{\sqrt{\sum \|g_j\|^2} + \epsilon}\right)$ across multi-tensor parameter sets.
5. **StateDict Round-Trip**: Perfect serialization / deserialization of optimizer moment buffers.
