# `brain-dataset` (v0.2.0)

> Comprehensive Dataset Ecosystem for Vision, Audio, Text, and Tabular Modalities.

## Overview

`brain-dataset` provides ready-to-use synthetic and standard dataset generators, split managers, transform pipelines, and caching abstractions across vision, natural language, audio, and tabular domains.

## Architecture

| Module | Description |
|---|---|
| `vision` | MNIST, CIFAR-10, ImageNet-style synthetic generators, and image folder loaders |
| `text` | Language modeling datasets, tokenized text iterators, paired translation datasets |
| `audio` | Speech command datasets, sound classification loaders, paired audio-text corpora |
| `tabular` | CSV parsing, normalization, categorical one-hot encoding, and feature scaling |
| `splits` | Train/validation/test splitters with k-fold cross validation and stratification |

## Quality & Verification

- **Tests**: 14,679 passed · 0 failed · 0 ignored
- **Clippy**: Clean (`cargo clippy -p brain-dataset -- -D warnings`)
- **Dependencies**: `std` + `brain-core`
