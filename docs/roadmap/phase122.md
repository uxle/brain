# Phase 122: PixelShuffle & PixelUnshuffle

**Stage:** E — Neural Network Layers, Normalizations & Activations
**Status:** ✅ Complete
**Pass Rate:** Verified in `nn_layers_harness.rs`

## Objective
Verify sub-pixel spatial rearrangement for image super-resolution: $[C \cdot r^2, H, W] \leftrightarrow [C, H \cdot r, W \cdot r]$.
