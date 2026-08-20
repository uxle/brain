//! # Video & HDMI Capture Abstraction (V4L2 / UVC / Mock)
//!
//! Provides raw and tensor frame streaming from camera, screen, and HDMI capture cards.

use brain_core::Tensor;
use std::sync::{Arc, Mutex};

/// Video Frame representation (RGB8 format).
#[derive(Debug, Clone)]
pub struct VideoFrame {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub data: Vec<u8>,
    pub timestamp_us: u64,
}

impl VideoFrame {
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Self {
        Self {
            width,
            height,
            channels: 3,
            data,
            timestamp_us: 0,
        }
    }

    /// Converts normalized [0.0, 1.0] RGB frame into a Tensor of shape [1, 3, height, width].
    pub fn to_tensor(&self) -> Tensor {
        let numel = self.width * self.height;
        let mut r = Vec::with_capacity(numel);
        let mut g = Vec::with_capacity(numel);
        let mut b = Vec::with_capacity(numel);

        for chunk in self.data.chunks_exact(3) {
            r.push(chunk[0] as f64 / 255.0);
            g.push(chunk[1] as f64 / 255.0);
            b.push(chunk[2] as f64 / 255.0);
        }

        let mut out = Vec::with_capacity(numel * 3);
        out.extend(r);
        out.extend(g);
        out.extend(b);

        Tensor::from_vec(out, vec![1, 3, self.height, self.width])
    }
}

/// Abstract Video Source Trait.
pub trait VideoSource: Send + Sync {
    /// Captures the latest available video frame.
    fn capture_frame(&self) -> Result<VideoFrame, String>;
}

/// Mock Video Source generating synthetic test frames.
#[derive(Debug, Clone)]
pub struct MockVideoSource {
    pub width: usize,
    pub height: usize,
    pub frame_counter: Arc<Mutex<usize>>,
}

impl MockVideoSource {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            frame_counter: Arc::new(Mutex::new(0)),
        }
    }
}

impl VideoSource for MockVideoSource {
    fn capture_frame(&self) -> Result<VideoFrame, String> {
        let mut cnt = self.frame_counter.lock().unwrap();
        *cnt += 1;
        let c = (*cnt % 255) as u8;

        // Generate solid color + pattern frame
        let total_bytes = self.width * self.height * 3;
        let mut data = vec![c; total_bytes];

        // Draw a distinct pixel box
        if self.width >= 10 && self.height >= 10 {
            for y in 0..10 {
                for x in 0..10 {
                    let idx = (y * self.width + x) * 3;
                    data[idx] = 255;
                    data[idx + 1] = 0;
                    data[idx + 2] = 0;
                }
            }
        }

        Ok(VideoFrame::new(self.width, self.height, data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_video_to_tensor() {
        let source = MockVideoSource::new(32, 32);
        let frame = source.capture_frame().unwrap();
        assert_eq!(frame.width, 32);
        assert_eq!(frame.height, 32);

        let t = frame.to_tensor();
        assert_eq!(t.shape(), &[1, 3, 32, 32]);
        assert!(t.data()[0] >= 0.0 && t.data()[0] <= 1.0);
    }
}
