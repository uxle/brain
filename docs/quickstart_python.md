# Brain Framework 10-Minute Quickstart (Python)

Welcome to the Python API for the **Brain** framework — pure-Rust performance with Python ergonomics.

---

## 1. Installation

Build and install locally via `maturin`:
```bash
maturin develop --release -m crates/brain-python/Cargo.toml
```

Or install the pre-built wheel:
```bash
pip install brain-dl
```

---

## 2. Basic Tensor Operations

```python
import brain

# Create tensors
x = brain.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
y = brain.ones([2, 2])

# Matrix multiplication & arithmetic
z = x @ y + x
print("Shape:", z.shape)
print("Data:", z.to_list())
```

---

## 3. Building & Training a Model

```python
import brain
import brain.nn as nn
import brain.optim as optim

# Define linear layer
layer = nn.Linear(in_features=4, out_features=2, bias=True)
optimizer = optim.AdamW(lr=1e-3, weight_decay=0.01)

# Forward pass
inputs = brain.ones([1, 4])
outputs = layer.forward(inputs)
print("Outputs:", outputs.to_list())
```
