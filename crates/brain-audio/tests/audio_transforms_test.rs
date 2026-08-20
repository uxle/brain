//! Tests for Audio transforms and STFT
use brain_audio::prelude::*;

#[test]
fn test_audio_buffer_creation() {
    let buf = AudioBuffer::from_mono(vec![0.0; 16000], SampleRate::SPEECH_16K).unwrap();
    assert_eq!(buf.num_samples(), 16000);
}

#[test]
fn test_stft_spectrogram_computation() {
    let cfg = STFTConfig::default();
    let buf = AudioBuffer::from_mono(vec![0.0; 1600], SampleRate::SPEECH_16K).unwrap();
    let spec = stft(&buf, &cfg);
    assert!(spec.is_ok());
}
