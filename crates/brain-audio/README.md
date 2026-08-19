# `brain-audio`

High-performance audio processing framework — DSP, spectrograms, features, augmentation, I/O, and VAD — in 100% safe, zero-dependency Rust.

## Overview

`brain-audio` provides production-grade audio pipelines for the Brain ecosystem on top of `brain-core` tensors: STFT/ISTFT and inverse transforms, log-Mel filterbanks, MFCC, chroma and wavelet features, pitch/energy/rhythm estimators, Griffin-Lim-style phase reconstruction, and window/frequency-scale utilities (Hann/Hamming/Blackman, Mel/Bark/ERB). It adds time-domain and SpecAugment-style augmentations, WAV/MP3/FLAC I/O, sinc/cubic/linear resampling, energy- and spectral-based voice activity detection, DTW alignment, and spectral-subtraction/Wiener denoising.

## Features

- **Core & features**: `AudioBuffer` (multi-channel, `from_mono`/`from_slice`/`from_tensor`), `SampleRate`, `Channels`, `AudioFormat`; `stft`, `istft`, `spectrogram`, `mel_spectrogram`, `mfcc`, chroma/wavelets in `feature/`.
- **Configs**: `STFTConfig` (`default_speech`, `default_music`), `MelConfig`, `MFCCConfig`, `WindowType` — all validated.
- **DSP ops**: `pre_emphasis`/`de_emphasis`, `hilbert_envelope`, `real_cepstrum`, `filter_2d`.
- **Augmentation**: `time_stretch`, `pitch_shift`, `spec_augment`, noise/gain/fade/clip distortions, frequency & time masking.
- **I/O & resample**: `read_wav`/`write_wav`, MP3 frame parsing, FLAC metadata parsing, `resample_audio`/`resample_1d` (`ResampleMethod`).
- **Analysis**: `compute_vad`/`trim_silence`, `dynamic_time_warping`, `spectral_subtraction`, pitch (`pitch_autocorr`), energy (`rms_frames`, `envelope_follower`), rhythm (`estimate_tempo`), and audio compression codecs (`mu_law`, `a_law`).
- **Batching**: `collate_audio_batch` with padding modes.

## Modules

| Module | Contents |
|---|---|
| `core.rs` | `AudioBuffer`, `SampleRate`, `Channels`, `AudioFormat` |
| `config.rs` | `STFTConfig`, `MelConfig`, `MFCCConfig`, `WindowType` |
| `feature/` | `stft`, `spectral`, `mfcc`, `tonal` (chroma), `wavelet` |
| `augment/` | `time`, `spectral`, `effects` |
| `io/` | `wav`, `mp3`, `flac` |
| `resample.rs` | Sinc/cubic/linear SRC (`ResampleMethod`) |
| `vad.rs`, `align.rs`, `denoise.rs` | Energy/spectral VAD, DTW, spectral subtraction |
| `features_pitch.rs`, `features_energy.rs`, `features_rhythm.rs` | F0, RMS/envelope, tempo estimators |
| `encoding.rs` | μ-law / A-law codecs, phonetic one-hot |
| `batch.rs` | `collate_audio_batch` |
| `ops.rs`, `utils.rs` | DSP ops, windows, frequency scales |

## Quick Start

```rust
use brain_audio::prelude::*;

let sr = SampleRate::new(16_000).unwrap();
let audio = AudioBuffer::from_mono(vec![0.0; 1600], sr).unwrap();
let (magnitude, phase) = stft(&audio, &STFTConfig::default_speech()).unwrap();
println!("{:?}", magnitude.shape());
```

## Testing

```bash
cargo test -p brain-audio -j 2
```

## Workspace Role

Audio feature-extraction and DSP layer for the Brain ecosystem. Depends only on `brain-core` (tensors) — zero external dependencies, 100% safe Rust.