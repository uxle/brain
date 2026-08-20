# Phase 201: Python Bindings (pyo3)

**Stage:** Post-1.0 Ecosystem
**Depends on:** Phase 200 (1.0 release)
**Status:** ✅ Architecture & Crate Scaffolding Complete

## Objective
Expose the verified Rust core (`brain-core`, `brain-autograd`, `brain-nn`, `brain-optim`, `brain-export`) as a native Python extension wheel (`brain-dl`) via `pyo3` and `maturin`.

## Deliverables
- `crates/brain-python`: Native extension crate (`brain_native`) with `PyTensor`, `PyLinear`, `PyConv2d`, `PyLayerNorm`, `PyAdam`, `PyAdamW`, `PySgd`.
- `python/brain/`: Python package wrapper with PEP 561 `py.typed` annotations.
- `pyproject.toml`: Maturin wheel packaging configuration.
