# `brain-rnn`

Recurrent neural network framework — LSTM, GRU, vanilla RNN, attention, bidirectional, packed sequences, and beam search — in 100% safe, zero-dependency Rust.

## Overview

`brain-rnn` provides production-grade recurrent architectures for the Brain ecosystem: four cell families (`LstmCell`, `GruCell`, `VanillaRnnCell`, `PeepholeLstmCell`, plus `AttentionCell` and layer-normalized `NormLstmCell`), multi-layer sequence unrolling (`LstmSeq`, `GruSeq`, `VanillaRnnSeq`), and a `RnnSequence` trait unifying them. It also ships bidirectional processing, variable-length `PackedSequence` batching, attention over encoder outputs, beam-search decoding, teacher forcing, and TBPTT/streaming utilities.

## Features

- **Cells**: `LstmCell`, `GruCell`, `VanillaRnnCell`, `PeepholeLstmCell`, `AttentionCell`, `NormLstmCell`, all behind the `RnnCell` trait with `CellState` (`h`/`c`).
- **Sequence models**: `LstmSeq`, `GruSeq`, `VanillaRnnSeq` (multi-layer, via `RnnSequence::forward`), `BidirectionalRnn` (concat/sum/mean `BidirectionalMerge`), `PackedSequence` for variable-length batches.
- **Decoding & attention**: `SeqAttention` (dot/additive/scaled variants), `BeamSearch` with `BeamConfig`/`BeamHypothesis`, `TeacherForcer` with `TeacherSchedule` (linear/exponential decay).
- **Training & streaming**: `VariationalDropout` (locked masks), `OnlineRnnStreamer` (stateful chunked inference), `truncate_steps` (TBPTT), `init_lstm_weights` + `RnnInitConfig`, and the fluent `RnnBuilder`.
- **Core types**: `RnnState`, `SequenceOutput` (`output` + `final_state`), `RnnConfig`/`CellConfig`/`CellKind`, `RnnError`/`RnnResult`, plus `create_padding_mask` and `forward_lstm` convenience helpers.

## Modules

| Module | Contents |
|---|---|
| `cells/` | `lstm`, `gru`, `rnn`, `lstm_peephole`, `attention_cell`, `normalized` |
| `seq/` | `lstm_seq`, `gru_seq`, `rnn_seq`, `bidirectional`, `packed`, `attention`, `beam`, `teacher` |
| `process.rs` | `OnlineRnnStreamer` |
| `reg_ops.rs` | `VariationalDropout` |
| `backward_ops.rs` | `truncate_steps` (TBPTT) |
| `init_rnn.rs` | `RnnInitConfig`, `init_lstm_weights` |
| `builder.rs`, `helper.rs`, `core.rs`, `config.rs`, `ops.rs`, `utils.rs` | `RnnBuilder`, `create_padding_mask`, state/error types, configs, RNN ops, RNG |

## Quick Start

```rust
use brain_core::Tensor;
use brain_rnn::prelude::*;

// Build a 2-layer LSTM sequence model
let seq_model = RnnBuilder::new(32, 64)
    .lstm()
    .num_layers(2)
    .bidirectional(false)
    .build()?;

// Input: [batch=1, seq_len=10, input_dim=32]
let input = Tensor::from_slice(&vec![0.5; 320], vec![1, 10, 32]);
let output = seq_model.forward(&input, None)?;
println!("{:?}", output.output.shape());
```

## Testing

```bash
cargo test -p brain-rnn -j 2
```

## Workspace Role

Recurrent sequence modeling layer of the Brain stack. Depends on `brain-core` (tensors) and `brain-autograd` (gradients) — zero external dependencies, 100% safe Rust.