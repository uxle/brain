# `brain-rnn` — Recurrent Neural Network Framework

Production-grade recurrent neural network architectures, attention mechanisms, bidirectional sequence modeling, packed sequences, and beam search for the **Brain** deep learning framework.

## Features

- **Recurrent Cells**:
  - `LstmCell`: 4-gate LSTM with $+1.0$ forget-bias initialization for long-range temporal retention.
  - `GruCell`: 3-gate GRU with reset, update, and candidate activations.
  - `VanillaRnnCell`: Standard Elman recurrent cell with Tanh non-linearities.
  - `PeepholeLstmCell`: Direct cell-to-gate connections ($w_{ci}, w_{cf}, w_{co}$).
  - `AttentionCell`: Bahdanau-style input scoring before cell transitions.
  - `NormLstmCell`: Layer-Normalized LSTM (LN-LSTM) pre-activation stabilization.
- **Sequence Modeling**:
  - `LstmSeq`, `GruSeq`, `VanillaRnnSeq`: Multi-layer stacked recurrent sequence unrolling.
  - `BidirectionalRnn`: Forward and backward passes combined via Concatenation, Summation, or Average.
  - `PackedSequence`: Efficient variable-length batch computation and masking.
- **Decoding & Attention**:
  - `SeqAttention`: Global attention over encoder outputs (Dot, Additive, Scaled Dot-Product).
  - `BeamSearch`: Top-$k$ hypothesis decoding with length penalty.
  - `TeacherForcer`: Scheduled sampling with linear and exponential decay schedules.
- **Regularization & Processing**:
  - `VariationalDropout`: Recurrent locked dropout masks across timesteps.
  - `OnlineRnnStreamer`: Stateful streaming inference for chunked sequence evaluation.
  - `truncate_steps`: Truncated Backpropagation Through Time (TBPTT) windowing.
  - `RnnBuilder`: Ergonomic fluent builder pattern.

## Quick Start

```rust
use brain_core::Tensor;
use brain_rnn::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Build a 2-layer LSTM sequence model
    let seq_model = RnnBuilder::new(32, 64)
        .lstm()
        .num_layers(2)
        .bidirectional(false)
        .build()?;

    // 2. Input sequence tensor: [batch_size: 1, seq_len: 10, input_dim: 32]
    let input = Tensor::from_slice(&vec![0.5; 320], vec![1, 10, 32]);

    // 3. Forward pass across all timesteps
    let output = seq_model.forward(&input, None)?;
    println!("Sequence output shape: {:?}", output.output.shape());

    Ok(())
}
```
