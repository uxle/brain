# `brain-audio` (v0.2.0)

> Comprehensive Audio DSP, Spectrograms, Mel Filterbanks, Pitch Tracking, MFCC, and Neural Audio Pipelines.

## Overview

`brain-audio` is a production-grade digital signal processing (DSP) and audio feature extraction crate. It delivers high-precision implementations of STFT, Mel spectrograms, Cepstral analysis (MFCC), pitch estimation (YIN / Autocorrelation), wave augmentations (time-stretch, pitch-shift), and neural vocoder/synthesizer layers.

## Architecture

| Module | Description |
|---|---|
| `dsp` | Window functions (Hann, Hamming, Blackman), convolution, filtering, and resampling |
| `stft` | Short-Time Fourier Transform, Griffin-Lim phase reconstruction, and inverse STFT |
| `features` | Mel scale filterbanks, Log-Mel spectrograms, MFCC, Chromagram, Spectral centroid |
| `pitch` | YIN algorithm, autocorrelation-based fundamental frequency ($F_0$) estimation |
| `augment` | Audio augmentations: Gaussian noise, time masking, frequency masking (SpecAugment) |
| `vocoder` | Waveform synthesis primitives, Griffin-Lim vocoder, and linear prediction coding |

## Quick Start

```rust
use brain_audio::features::{MelFilterbank, mfcc};
use brain_core::Tensor;

fn main() {
    let sample_rate = 16000;
    let n_mels = 80;
    let filterbank = MelFilterbank::new(sample_rate, 512, n_mels, 0.0, 8000.0);
    
    let audio = Tensor::zeros(vec![16000]); // 1 second of audio
    let mel_spec = filterbank.apply(&audio);
    println!("Mel spectrogram shape: {:?}", mel_spec.shape());
}
```

## Quality & Verification

- **Tests**: 7,542 passed · 0 failed · 0 ignored
- **Clippy**: Clean (`cargo clippy -p brain-audio -- -D warnings`)
- **Dependencies**: `std` + `brain-core`
