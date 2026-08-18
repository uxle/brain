# Contributing to Brain

Thank you for your interest in contributing to the **Brain** deep learning framework.

Brain is built on the philosophy of 100% safe, pure-Rust code with zero external C++ or non-standard dependencies. We hold every component to rigorous standards of numerical correctness and mathematical verifiability.

---

## 1. Development & Resource Safety Guidelines

> [!CAUTION]
> **Strict Concurrency Rule**: When running `cargo build` or `cargo test`, **ALWAYS** pass `-j 2` (or `-j 1`).
> Never execute unconstrained full-workspace builds (`cargo test --workspace` across all 33 crates simultaneously) to avoid CPU/memory starvation on developer host machines.

Always target tests per crate:
```bash
cargo test -p brain-core --test numerical_check -j 2
cargo test -p brain-autograd --test grad_check -j 2
cargo test -p brain-train --test trainer_regression -j 2
```

---

## 2. Correctness Protocol for Adding Operations

When introducing a new neural operator, layer, or loss function:
1. Implement the forward math with checked shape calculations.
2. Implement exact analytical Vector-Jacobian Products (VJP) in `crates/brain-autograd/src/ops/`.
3. Add a finite-difference verification test in `crates/brain-autograd/tests/grad_check.rs` asserting numeric vs analytical difference $< 1e-4$.
4. Never return dummy zero gradients for trainable operations.

---

## 3. Pull Request Checklist

Before submitting a pull request:
- [ ] `./scripts/ci.sh` runs cleanly and all per-crate test suites pass.
- [ ] Code formatting is verified: `cargo fmt --check`.
- [ ] New trainable operations have analytical VJPs and finite-difference tests.
- [ ] No external dependencies are added to production crates.
