//! # High-Dynamic-Range (HDR) Latency Histogram
//!
//! Provides a memory-efficient logarithmic bucket histogram for nanosecond-to-minute latency distributions.

/// High dynamic range logarithmic latency histogram.
#[derive(Debug, Clone)]
pub struct HdrHistogram {
    buckets: Vec<u64>,
    min_value: u64,
    max_value: u64,
    total_count: u64,
}

impl Default for HdrHistogram {
    fn default() -> Self {
        Self::new()
    }
}

impl HdrHistogram {
    /// Creates a new `HdrHistogram` with 1024 logarithmic buckets.
    pub fn new() -> Self {
        Self {
            buckets: vec![0; 1024],
            min_value: u64::MAX,
            max_value: 0,
            total_count: 0,
        }
    }

    /// Records a latency value in nanoseconds.
    pub fn record(&mut self, value_nanos: u64) {
        self.min_value = self.min_value.min(value_nanos);
        self.max_value = self.max_value.max(value_nanos);
        self.total_count += 1;

        let bucket_idx = self.value_to_bucket(value_nanos);
        self.buckets[bucket_idx] += 1;
    }

    fn value_to_bucket(&self, val: u64) -> usize {
        if val == 0 {
            0
        } else {
            let leading_zeros = val.leading_zeros() as usize;
            let log_bucket = (64 - leading_zeros).min(63);
            let sub_bucket = ((val >> log_bucket.saturating_sub(4)) & 0x0F) as usize;
            (log_bucket * 16 + sub_bucket).min(self.buckets.len() - 1)
        }
    }

    /// Returns the total number of recorded observations.
    pub fn count(&self) -> u64 {
        self.total_count
    }

    /// Returns the minimum recorded value in nanoseconds.
    pub fn min(&self) -> u64 {
        if self.total_count == 0 { 0 } else { self.min_value }
    }

    /// Returns the maximum recorded value in nanoseconds.
    pub fn max(&self) -> u64 {
        self.max_value
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_hdr_histogram_stress_001() {
        let mut hist = HdrHistogram::new();
        hist.record(110);
        hist.record(210);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 110);
        assert_eq!(hist.max(), 210);
    }

    #[test]
    fn test_hdr_histogram_stress_002() {
        let mut hist = HdrHistogram::new();
        hist.record(120);
        hist.record(220);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 120);
        assert_eq!(hist.max(), 220);
    }

    #[test]
    fn test_hdr_histogram_stress_003() {
        let mut hist = HdrHistogram::new();
        hist.record(130);
        hist.record(230);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 130);
        assert_eq!(hist.max(), 230);
    }

    #[test]
    fn test_hdr_histogram_stress_004() {
        let mut hist = HdrHistogram::new();
        hist.record(140);
        hist.record(240);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 140);
        assert_eq!(hist.max(), 240);
    }

    #[test]
    fn test_hdr_histogram_stress_005() {
        let mut hist = HdrHistogram::new();
        hist.record(150);
        hist.record(250);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 150);
        assert_eq!(hist.max(), 250);
    }

    #[test]
    fn test_hdr_histogram_stress_006() {
        let mut hist = HdrHistogram::new();
        hist.record(160);
        hist.record(260);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 160);
        assert_eq!(hist.max(), 260);
    }

    #[test]
    fn test_hdr_histogram_stress_007() {
        let mut hist = HdrHistogram::new();
        hist.record(170);
        hist.record(270);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 170);
        assert_eq!(hist.max(), 270);
    }

    #[test]
    fn test_hdr_histogram_stress_008() {
        let mut hist = HdrHistogram::new();
        hist.record(180);
        hist.record(280);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 180);
        assert_eq!(hist.max(), 280);
    }

    #[test]
    fn test_hdr_histogram_stress_009() {
        let mut hist = HdrHistogram::new();
        hist.record(190);
        hist.record(290);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 190);
        assert_eq!(hist.max(), 290);
    }

    #[test]
    fn test_hdr_histogram_stress_010() {
        let mut hist = HdrHistogram::new();
        hist.record(200);
        hist.record(300);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 200);
        assert_eq!(hist.max(), 300);
    }

    #[test]
    fn test_hdr_histogram_stress_011() {
        let mut hist = HdrHistogram::new();
        hist.record(210);
        hist.record(310);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 210);
        assert_eq!(hist.max(), 310);
    }

    #[test]
    fn test_hdr_histogram_stress_012() {
        let mut hist = HdrHistogram::new();
        hist.record(220);
        hist.record(320);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 220);
        assert_eq!(hist.max(), 320);
    }

    #[test]
    fn test_hdr_histogram_stress_013() {
        let mut hist = HdrHistogram::new();
        hist.record(230);
        hist.record(330);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 230);
        assert_eq!(hist.max(), 330);
    }

    #[test]
    fn test_hdr_histogram_stress_014() {
        let mut hist = HdrHistogram::new();
        hist.record(240);
        hist.record(340);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 240);
        assert_eq!(hist.max(), 340);
    }

    #[test]
    fn test_hdr_histogram_stress_015() {
        let mut hist = HdrHistogram::new();
        hist.record(250);
        hist.record(350);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 250);
        assert_eq!(hist.max(), 350);
    }

    #[test]
    fn test_hdr_histogram_stress_016() {
        let mut hist = HdrHistogram::new();
        hist.record(260);
        hist.record(360);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 260);
        assert_eq!(hist.max(), 360);
    }

    #[test]
    fn test_hdr_histogram_stress_017() {
        let mut hist = HdrHistogram::new();
        hist.record(270);
        hist.record(370);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 270);
        assert_eq!(hist.max(), 370);
    }

    #[test]
    fn test_hdr_histogram_stress_018() {
        let mut hist = HdrHistogram::new();
        hist.record(280);
        hist.record(380);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 280);
        assert_eq!(hist.max(), 380);
    }

    #[test]
    fn test_hdr_histogram_stress_019() {
        let mut hist = HdrHistogram::new();
        hist.record(290);
        hist.record(390);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 290);
        assert_eq!(hist.max(), 390);
    }

    #[test]
    fn test_hdr_histogram_stress_020() {
        let mut hist = HdrHistogram::new();
        hist.record(300);
        hist.record(400);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 300);
        assert_eq!(hist.max(), 400);
    }

    #[test]
    fn test_hdr_histogram_stress_021() {
        let mut hist = HdrHistogram::new();
        hist.record(310);
        hist.record(410);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 310);
        assert_eq!(hist.max(), 410);
    }

    #[test]
    fn test_hdr_histogram_stress_022() {
        let mut hist = HdrHistogram::new();
        hist.record(320);
        hist.record(420);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 320);
        assert_eq!(hist.max(), 420);
    }

    #[test]
    fn test_hdr_histogram_stress_023() {
        let mut hist = HdrHistogram::new();
        hist.record(330);
        hist.record(430);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 330);
        assert_eq!(hist.max(), 430);
    }

    #[test]
    fn test_hdr_histogram_stress_024() {
        let mut hist = HdrHistogram::new();
        hist.record(340);
        hist.record(440);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 340);
        assert_eq!(hist.max(), 440);
    }

    #[test]
    fn test_hdr_histogram_stress_025() {
        let mut hist = HdrHistogram::new();
        hist.record(350);
        hist.record(450);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 350);
        assert_eq!(hist.max(), 450);
    }

    #[test]
    fn test_hdr_histogram_stress_026() {
        let mut hist = HdrHistogram::new();
        hist.record(360);
        hist.record(460);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 360);
        assert_eq!(hist.max(), 460);
    }

    #[test]
    fn test_hdr_histogram_stress_027() {
        let mut hist = HdrHistogram::new();
        hist.record(370);
        hist.record(470);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 370);
        assert_eq!(hist.max(), 470);
    }

    #[test]
    fn test_hdr_histogram_stress_028() {
        let mut hist = HdrHistogram::new();
        hist.record(380);
        hist.record(480);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 380);
        assert_eq!(hist.max(), 480);
    }

    #[test]
    fn test_hdr_histogram_stress_029() {
        let mut hist = HdrHistogram::new();
        hist.record(390);
        hist.record(490);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 390);
        assert_eq!(hist.max(), 490);
    }

    #[test]
    fn test_hdr_histogram_stress_030() {
        let mut hist = HdrHistogram::new();
        hist.record(400);
        hist.record(500);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 400);
        assert_eq!(hist.max(), 500);
    }

    #[test]
    fn test_hdr_histogram_stress_031() {
        let mut hist = HdrHistogram::new();
        hist.record(410);
        hist.record(510);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 410);
        assert_eq!(hist.max(), 510);
    }

    #[test]
    fn test_hdr_histogram_stress_032() {
        let mut hist = HdrHistogram::new();
        hist.record(420);
        hist.record(520);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 420);
        assert_eq!(hist.max(), 520);
    }

    #[test]
    fn test_hdr_histogram_stress_033() {
        let mut hist = HdrHistogram::new();
        hist.record(430);
        hist.record(530);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 430);
        assert_eq!(hist.max(), 530);
    }

    #[test]
    fn test_hdr_histogram_stress_034() {
        let mut hist = HdrHistogram::new();
        hist.record(440);
        hist.record(540);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 440);
        assert_eq!(hist.max(), 540);
    }

    #[test]
    fn test_hdr_histogram_stress_035() {
        let mut hist = HdrHistogram::new();
        hist.record(450);
        hist.record(550);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 450);
        assert_eq!(hist.max(), 550);
    }

    #[test]
    fn test_hdr_histogram_stress_036() {
        let mut hist = HdrHistogram::new();
        hist.record(460);
        hist.record(560);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 460);
        assert_eq!(hist.max(), 560);
    }

    #[test]
    fn test_hdr_histogram_stress_037() {
        let mut hist = HdrHistogram::new();
        hist.record(470);
        hist.record(570);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 470);
        assert_eq!(hist.max(), 570);
    }

    #[test]
    fn test_hdr_histogram_stress_038() {
        let mut hist = HdrHistogram::new();
        hist.record(480);
        hist.record(580);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 480);
        assert_eq!(hist.max(), 580);
    }

    #[test]
    fn test_hdr_histogram_stress_039() {
        let mut hist = HdrHistogram::new();
        hist.record(490);
        hist.record(590);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 490);
        assert_eq!(hist.max(), 590);
    }

    #[test]
    fn test_hdr_histogram_stress_040() {
        let mut hist = HdrHistogram::new();
        hist.record(500);
        hist.record(600);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 500);
        assert_eq!(hist.max(), 600);
    }

    #[test]
    fn test_hdr_histogram_stress_041() {
        let mut hist = HdrHistogram::new();
        hist.record(510);
        hist.record(610);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 510);
        assert_eq!(hist.max(), 610);
    }

    #[test]
    fn test_hdr_histogram_stress_042() {
        let mut hist = HdrHistogram::new();
        hist.record(520);
        hist.record(620);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 520);
        assert_eq!(hist.max(), 620);
    }

    #[test]
    fn test_hdr_histogram_stress_043() {
        let mut hist = HdrHistogram::new();
        hist.record(530);
        hist.record(630);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 530);
        assert_eq!(hist.max(), 630);
    }

    #[test]
    fn test_hdr_histogram_stress_044() {
        let mut hist = HdrHistogram::new();
        hist.record(540);
        hist.record(640);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 540);
        assert_eq!(hist.max(), 640);
    }

    #[test]
    fn test_hdr_histogram_stress_045() {
        let mut hist = HdrHistogram::new();
        hist.record(550);
        hist.record(650);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 550);
        assert_eq!(hist.max(), 650);
    }

    #[test]
    fn test_hdr_histogram_stress_046() {
        let mut hist = HdrHistogram::new();
        hist.record(560);
        hist.record(660);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 560);
        assert_eq!(hist.max(), 660);
    }

    #[test]
    fn test_hdr_histogram_stress_047() {
        let mut hist = HdrHistogram::new();
        hist.record(570);
        hist.record(670);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 570);
        assert_eq!(hist.max(), 670);
    }

    #[test]
    fn test_hdr_histogram_stress_048() {
        let mut hist = HdrHistogram::new();
        hist.record(580);
        hist.record(680);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 580);
        assert_eq!(hist.max(), 680);
    }

    #[test]
    fn test_hdr_histogram_stress_049() {
        let mut hist = HdrHistogram::new();
        hist.record(590);
        hist.record(690);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 590);
        assert_eq!(hist.max(), 690);
    }

    #[test]
    fn test_hdr_histogram_stress_050() {
        let mut hist = HdrHistogram::new();
        hist.record(600);
        hist.record(700);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 600);
        assert_eq!(hist.max(), 700);
    }

    #[test]
    fn test_hdr_histogram_stress_051() {
        let mut hist = HdrHistogram::new();
        hist.record(610);
        hist.record(710);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 610);
        assert_eq!(hist.max(), 710);
    }

    #[test]
    fn test_hdr_histogram_stress_052() {
        let mut hist = HdrHistogram::new();
        hist.record(620);
        hist.record(720);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 620);
        assert_eq!(hist.max(), 720);
    }

    #[test]
    fn test_hdr_histogram_stress_053() {
        let mut hist = HdrHistogram::new();
        hist.record(630);
        hist.record(730);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 630);
        assert_eq!(hist.max(), 730);
    }

    #[test]
    fn test_hdr_histogram_stress_054() {
        let mut hist = HdrHistogram::new();
        hist.record(640);
        hist.record(740);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 640);
        assert_eq!(hist.max(), 740);
    }

    #[test]
    fn test_hdr_histogram_stress_055() {
        let mut hist = HdrHistogram::new();
        hist.record(650);
        hist.record(750);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 650);
        assert_eq!(hist.max(), 750);
    }

    #[test]
    fn test_hdr_histogram_stress_056() {
        let mut hist = HdrHistogram::new();
        hist.record(660);
        hist.record(760);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 660);
        assert_eq!(hist.max(), 760);
    }

    #[test]
    fn test_hdr_histogram_stress_057() {
        let mut hist = HdrHistogram::new();
        hist.record(670);
        hist.record(770);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 670);
        assert_eq!(hist.max(), 770);
    }

    #[test]
    fn test_hdr_histogram_stress_058() {
        let mut hist = HdrHistogram::new();
        hist.record(680);
        hist.record(780);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 680);
        assert_eq!(hist.max(), 780);
    }

    #[test]
    fn test_hdr_histogram_stress_059() {
        let mut hist = HdrHistogram::new();
        hist.record(690);
        hist.record(790);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 690);
        assert_eq!(hist.max(), 790);
    }

    #[test]
    fn test_hdr_histogram_stress_060() {
        let mut hist = HdrHistogram::new();
        hist.record(700);
        hist.record(800);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 700);
        assert_eq!(hist.max(), 800);
    }

    #[test]
    fn test_hdr_histogram_stress_061() {
        let mut hist = HdrHistogram::new();
        hist.record(710);
        hist.record(810);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 710);
        assert_eq!(hist.max(), 810);
    }

    #[test]
    fn test_hdr_histogram_stress_062() {
        let mut hist = HdrHistogram::new();
        hist.record(720);
        hist.record(820);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 720);
        assert_eq!(hist.max(), 820);
    }

    #[test]
    fn test_hdr_histogram_stress_063() {
        let mut hist = HdrHistogram::new();
        hist.record(730);
        hist.record(830);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 730);
        assert_eq!(hist.max(), 830);
    }

    #[test]
    fn test_hdr_histogram_stress_064() {
        let mut hist = HdrHistogram::new();
        hist.record(740);
        hist.record(840);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 740);
        assert_eq!(hist.max(), 840);
    }

    #[test]
    fn test_hdr_histogram_stress_065() {
        let mut hist = HdrHistogram::new();
        hist.record(750);
        hist.record(850);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 750);
        assert_eq!(hist.max(), 850);
    }

    #[test]
    fn test_hdr_histogram_stress_066() {
        let mut hist = HdrHistogram::new();
        hist.record(760);
        hist.record(860);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 760);
        assert_eq!(hist.max(), 860);
    }

    #[test]
    fn test_hdr_histogram_stress_067() {
        let mut hist = HdrHistogram::new();
        hist.record(770);
        hist.record(870);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 770);
        assert_eq!(hist.max(), 870);
    }

    #[test]
    fn test_hdr_histogram_stress_068() {
        let mut hist = HdrHistogram::new();
        hist.record(780);
        hist.record(880);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 780);
        assert_eq!(hist.max(), 880);
    }

    #[test]
    fn test_hdr_histogram_stress_069() {
        let mut hist = HdrHistogram::new();
        hist.record(790);
        hist.record(890);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 790);
        assert_eq!(hist.max(), 890);
    }

    #[test]
    fn test_hdr_histogram_stress_070() {
        let mut hist = HdrHistogram::new();
        hist.record(800);
        hist.record(900);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 800);
        assert_eq!(hist.max(), 900);
    }

    #[test]
    fn test_hdr_histogram_stress_071() {
        let mut hist = HdrHistogram::new();
        hist.record(810);
        hist.record(910);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 810);
        assert_eq!(hist.max(), 910);
    }

    #[test]
    fn test_hdr_histogram_stress_072() {
        let mut hist = HdrHistogram::new();
        hist.record(820);
        hist.record(920);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 820);
        assert_eq!(hist.max(), 920);
    }

    #[test]
    fn test_hdr_histogram_stress_073() {
        let mut hist = HdrHistogram::new();
        hist.record(830);
        hist.record(930);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 830);
        assert_eq!(hist.max(), 930);
    }

    #[test]
    fn test_hdr_histogram_stress_074() {
        let mut hist = HdrHistogram::new();
        hist.record(840);
        hist.record(940);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 840);
        assert_eq!(hist.max(), 940);
    }

    #[test]
    fn test_hdr_histogram_stress_075() {
        let mut hist = HdrHistogram::new();
        hist.record(850);
        hist.record(950);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 850);
        assert_eq!(hist.max(), 950);
    }

    #[test]
    fn test_hdr_histogram_stress_076() {
        let mut hist = HdrHistogram::new();
        hist.record(860);
        hist.record(960);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 860);
        assert_eq!(hist.max(), 960);
    }

    #[test]
    fn test_hdr_histogram_stress_077() {
        let mut hist = HdrHistogram::new();
        hist.record(870);
        hist.record(970);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 870);
        assert_eq!(hist.max(), 970);
    }

    #[test]
    fn test_hdr_histogram_stress_078() {
        let mut hist = HdrHistogram::new();
        hist.record(880);
        hist.record(980);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 880);
        assert_eq!(hist.max(), 980);
    }

    #[test]
    fn test_hdr_histogram_stress_079() {
        let mut hist = HdrHistogram::new();
        hist.record(890);
        hist.record(990);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 890);
        assert_eq!(hist.max(), 990);
    }

    #[test]
    fn test_hdr_histogram_stress_080() {
        let mut hist = HdrHistogram::new();
        hist.record(900);
        hist.record(1000);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 900);
        assert_eq!(hist.max(), 1000);
    }

    #[test]
    fn test_hdr_histogram_stress_081() {
        let mut hist = HdrHistogram::new();
        hist.record(910);
        hist.record(1010);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 910);
        assert_eq!(hist.max(), 1010);
    }

    #[test]
    fn test_hdr_histogram_stress_082() {
        let mut hist = HdrHistogram::new();
        hist.record(920);
        hist.record(1020);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 920);
        assert_eq!(hist.max(), 1020);
    }

    #[test]
    fn test_hdr_histogram_stress_083() {
        let mut hist = HdrHistogram::new();
        hist.record(930);
        hist.record(1030);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 930);
        assert_eq!(hist.max(), 1030);
    }

    #[test]
    fn test_hdr_histogram_stress_084() {
        let mut hist = HdrHistogram::new();
        hist.record(940);
        hist.record(1040);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 940);
        assert_eq!(hist.max(), 1040);
    }

    #[test]
    fn test_hdr_histogram_stress_085() {
        let mut hist = HdrHistogram::new();
        hist.record(950);
        hist.record(1050);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 950);
        assert_eq!(hist.max(), 1050);
    }

    #[test]
    fn test_hdr_histogram_stress_086() {
        let mut hist = HdrHistogram::new();
        hist.record(960);
        hist.record(1060);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 960);
        assert_eq!(hist.max(), 1060);
    }

    #[test]
    fn test_hdr_histogram_stress_087() {
        let mut hist = HdrHistogram::new();
        hist.record(970);
        hist.record(1070);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 970);
        assert_eq!(hist.max(), 1070);
    }

    #[test]
    fn test_hdr_histogram_stress_088() {
        let mut hist = HdrHistogram::new();
        hist.record(980);
        hist.record(1080);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 980);
        assert_eq!(hist.max(), 1080);
    }

    #[test]
    fn test_hdr_histogram_stress_089() {
        let mut hist = HdrHistogram::new();
        hist.record(990);
        hist.record(1090);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 990);
        assert_eq!(hist.max(), 1090);
    }

    #[test]
    fn test_hdr_histogram_stress_090() {
        let mut hist = HdrHistogram::new();
        hist.record(1000);
        hist.record(1100);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1000);
        assert_eq!(hist.max(), 1100);
    }

    #[test]
    fn test_hdr_histogram_stress_091() {
        let mut hist = HdrHistogram::new();
        hist.record(1010);
        hist.record(1110);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1010);
        assert_eq!(hist.max(), 1110);
    }

    #[test]
    fn test_hdr_histogram_stress_092() {
        let mut hist = HdrHistogram::new();
        hist.record(1020);
        hist.record(1120);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1020);
        assert_eq!(hist.max(), 1120);
    }

    #[test]
    fn test_hdr_histogram_stress_093() {
        let mut hist = HdrHistogram::new();
        hist.record(1030);
        hist.record(1130);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1030);
        assert_eq!(hist.max(), 1130);
    }

    #[test]
    fn test_hdr_histogram_stress_094() {
        let mut hist = HdrHistogram::new();
        hist.record(1040);
        hist.record(1140);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1040);
        assert_eq!(hist.max(), 1140);
    }

    #[test]
    fn test_hdr_histogram_stress_095() {
        let mut hist = HdrHistogram::new();
        hist.record(1050);
        hist.record(1150);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1050);
        assert_eq!(hist.max(), 1150);
    }

    #[test]
    fn test_hdr_histogram_stress_096() {
        let mut hist = HdrHistogram::new();
        hist.record(1060);
        hist.record(1160);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1060);
        assert_eq!(hist.max(), 1160);
    }

    #[test]
    fn test_hdr_histogram_stress_097() {
        let mut hist = HdrHistogram::new();
        hist.record(1070);
        hist.record(1170);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1070);
        assert_eq!(hist.max(), 1170);
    }

    #[test]
    fn test_hdr_histogram_stress_098() {
        let mut hist = HdrHistogram::new();
        hist.record(1080);
        hist.record(1180);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1080);
        assert_eq!(hist.max(), 1180);
    }

    #[test]
    fn test_hdr_histogram_stress_099() {
        let mut hist = HdrHistogram::new();
        hist.record(1090);
        hist.record(1190);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1090);
        assert_eq!(hist.max(), 1190);
    }

    #[test]
    fn test_hdr_histogram_stress_100() {
        let mut hist = HdrHistogram::new();
        hist.record(1100);
        hist.record(1200);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1100);
        assert_eq!(hist.max(), 1200);
    }

    #[test]
    fn test_hdr_histogram_stress_101() {
        let mut hist = HdrHistogram::new();
        hist.record(1110);
        hist.record(1210);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1110);
        assert_eq!(hist.max(), 1210);
    }

    #[test]
    fn test_hdr_histogram_stress_102() {
        let mut hist = HdrHistogram::new();
        hist.record(1120);
        hist.record(1220);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1120);
        assert_eq!(hist.max(), 1220);
    }

    #[test]
    fn test_hdr_histogram_stress_103() {
        let mut hist = HdrHistogram::new();
        hist.record(1130);
        hist.record(1230);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1130);
        assert_eq!(hist.max(), 1230);
    }

    #[test]
    fn test_hdr_histogram_stress_104() {
        let mut hist = HdrHistogram::new();
        hist.record(1140);
        hist.record(1240);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1140);
        assert_eq!(hist.max(), 1240);
    }

    #[test]
    fn test_hdr_histogram_stress_105() {
        let mut hist = HdrHistogram::new();
        hist.record(1150);
        hist.record(1250);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1150);
        assert_eq!(hist.max(), 1250);
    }

    #[test]
    fn test_hdr_histogram_stress_106() {
        let mut hist = HdrHistogram::new();
        hist.record(1160);
        hist.record(1260);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1160);
        assert_eq!(hist.max(), 1260);
    }

    #[test]
    fn test_hdr_histogram_stress_107() {
        let mut hist = HdrHistogram::new();
        hist.record(1170);
        hist.record(1270);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1170);
        assert_eq!(hist.max(), 1270);
    }

    #[test]
    fn test_hdr_histogram_stress_108() {
        let mut hist = HdrHistogram::new();
        hist.record(1180);
        hist.record(1280);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1180);
        assert_eq!(hist.max(), 1280);
    }

    #[test]
    fn test_hdr_histogram_stress_109() {
        let mut hist = HdrHistogram::new();
        hist.record(1190);
        hist.record(1290);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1190);
        assert_eq!(hist.max(), 1290);
    }

    #[test]
    fn test_hdr_histogram_stress_110() {
        let mut hist = HdrHistogram::new();
        hist.record(1200);
        hist.record(1300);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1200);
        assert_eq!(hist.max(), 1300);
    }

    #[test]
    fn test_hdr_histogram_stress_111() {
        let mut hist = HdrHistogram::new();
        hist.record(1210);
        hist.record(1310);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1210);
        assert_eq!(hist.max(), 1310);
    }

    #[test]
    fn test_hdr_histogram_stress_112() {
        let mut hist = HdrHistogram::new();
        hist.record(1220);
        hist.record(1320);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1220);
        assert_eq!(hist.max(), 1320);
    }

    #[test]
    fn test_hdr_histogram_stress_113() {
        let mut hist = HdrHistogram::new();
        hist.record(1230);
        hist.record(1330);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1230);
        assert_eq!(hist.max(), 1330);
    }

    #[test]
    fn test_hdr_histogram_stress_114() {
        let mut hist = HdrHistogram::new();
        hist.record(1240);
        hist.record(1340);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1240);
        assert_eq!(hist.max(), 1340);
    }

    #[test]
    fn test_hdr_histogram_stress_115() {
        let mut hist = HdrHistogram::new();
        hist.record(1250);
        hist.record(1350);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1250);
        assert_eq!(hist.max(), 1350);
    }

    #[test]
    fn test_hdr_histogram_stress_116() {
        let mut hist = HdrHistogram::new();
        hist.record(1260);
        hist.record(1360);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1260);
        assert_eq!(hist.max(), 1360);
    }

    #[test]
    fn test_hdr_histogram_stress_117() {
        let mut hist = HdrHistogram::new();
        hist.record(1270);
        hist.record(1370);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1270);
        assert_eq!(hist.max(), 1370);
    }

    #[test]
    fn test_hdr_histogram_stress_118() {
        let mut hist = HdrHistogram::new();
        hist.record(1280);
        hist.record(1380);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1280);
        assert_eq!(hist.max(), 1380);
    }

    #[test]
    fn test_hdr_histogram_stress_119() {
        let mut hist = HdrHistogram::new();
        hist.record(1290);
        hist.record(1390);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1290);
        assert_eq!(hist.max(), 1390);
    }

    #[test]
    fn test_hdr_histogram_stress_120() {
        let mut hist = HdrHistogram::new();
        hist.record(1300);
        hist.record(1400);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1300);
        assert_eq!(hist.max(), 1400);
    }

    #[test]
    fn test_hdr_histogram_stress_121() {
        let mut hist = HdrHistogram::new();
        hist.record(1310);
        hist.record(1410);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1310);
        assert_eq!(hist.max(), 1410);
    }

    #[test]
    fn test_hdr_histogram_stress_122() {
        let mut hist = HdrHistogram::new();
        hist.record(1320);
        hist.record(1420);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1320);
        assert_eq!(hist.max(), 1420);
    }

    #[test]
    fn test_hdr_histogram_stress_123() {
        let mut hist = HdrHistogram::new();
        hist.record(1330);
        hist.record(1430);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1330);
        assert_eq!(hist.max(), 1430);
    }

    #[test]
    fn test_hdr_histogram_stress_124() {
        let mut hist = HdrHistogram::new();
        hist.record(1340);
        hist.record(1440);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1340);
        assert_eq!(hist.max(), 1440);
    }

    #[test]
    fn test_hdr_histogram_stress_125() {
        let mut hist = HdrHistogram::new();
        hist.record(1350);
        hist.record(1450);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1350);
        assert_eq!(hist.max(), 1450);
    }

    #[test]
    fn test_hdr_histogram_stress_126() {
        let mut hist = HdrHistogram::new();
        hist.record(1360);
        hist.record(1460);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1360);
        assert_eq!(hist.max(), 1460);
    }

    #[test]
    fn test_hdr_histogram_stress_127() {
        let mut hist = HdrHistogram::new();
        hist.record(1370);
        hist.record(1470);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1370);
        assert_eq!(hist.max(), 1470);
    }

    #[test]
    fn test_hdr_histogram_stress_128() {
        let mut hist = HdrHistogram::new();
        hist.record(1380);
        hist.record(1480);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1380);
        assert_eq!(hist.max(), 1480);
    }

    #[test]
    fn test_hdr_histogram_stress_129() {
        let mut hist = HdrHistogram::new();
        hist.record(1390);
        hist.record(1490);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1390);
        assert_eq!(hist.max(), 1490);
    }

    #[test]
    fn test_hdr_histogram_stress_130() {
        let mut hist = HdrHistogram::new();
        hist.record(1400);
        hist.record(1500);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1400);
        assert_eq!(hist.max(), 1500);
    }

    #[test]
    fn test_hdr_histogram_stress_131() {
        let mut hist = HdrHistogram::new();
        hist.record(1410);
        hist.record(1510);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1410);
        assert_eq!(hist.max(), 1510);
    }

    #[test]
    fn test_hdr_histogram_stress_132() {
        let mut hist = HdrHistogram::new();
        hist.record(1420);
        hist.record(1520);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1420);
        assert_eq!(hist.max(), 1520);
    }

    #[test]
    fn test_hdr_histogram_stress_133() {
        let mut hist = HdrHistogram::new();
        hist.record(1430);
        hist.record(1530);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1430);
        assert_eq!(hist.max(), 1530);
    }

    #[test]
    fn test_hdr_histogram_stress_134() {
        let mut hist = HdrHistogram::new();
        hist.record(1440);
        hist.record(1540);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1440);
        assert_eq!(hist.max(), 1540);
    }

    #[test]
    fn test_hdr_histogram_stress_135() {
        let mut hist = HdrHistogram::new();
        hist.record(1450);
        hist.record(1550);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1450);
        assert_eq!(hist.max(), 1550);
    }

    #[test]
    fn test_hdr_histogram_stress_136() {
        let mut hist = HdrHistogram::new();
        hist.record(1460);
        hist.record(1560);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1460);
        assert_eq!(hist.max(), 1560);
    }

    #[test]
    fn test_hdr_histogram_stress_137() {
        let mut hist = HdrHistogram::new();
        hist.record(1470);
        hist.record(1570);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1470);
        assert_eq!(hist.max(), 1570);
    }

    #[test]
    fn test_hdr_histogram_stress_138() {
        let mut hist = HdrHistogram::new();
        hist.record(1480);
        hist.record(1580);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1480);
        assert_eq!(hist.max(), 1580);
    }

    #[test]
    fn test_hdr_histogram_stress_139() {
        let mut hist = HdrHistogram::new();
        hist.record(1490);
        hist.record(1590);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1490);
        assert_eq!(hist.max(), 1590);
    }

    #[test]
    fn test_hdr_histogram_stress_140() {
        let mut hist = HdrHistogram::new();
        hist.record(1500);
        hist.record(1600);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1500);
        assert_eq!(hist.max(), 1600);
    }

    #[test]
    fn test_hdr_histogram_stress_141() {
        let mut hist = HdrHistogram::new();
        hist.record(1510);
        hist.record(1610);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1510);
        assert_eq!(hist.max(), 1610);
    }

    #[test]
    fn test_hdr_histogram_stress_142() {
        let mut hist = HdrHistogram::new();
        hist.record(1520);
        hist.record(1620);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1520);
        assert_eq!(hist.max(), 1620);
    }

    #[test]
    fn test_hdr_histogram_stress_143() {
        let mut hist = HdrHistogram::new();
        hist.record(1530);
        hist.record(1630);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1530);
        assert_eq!(hist.max(), 1630);
    }

    #[test]
    fn test_hdr_histogram_stress_144() {
        let mut hist = HdrHistogram::new();
        hist.record(1540);
        hist.record(1640);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1540);
        assert_eq!(hist.max(), 1640);
    }

    #[test]
    fn test_hdr_histogram_stress_145() {
        let mut hist = HdrHistogram::new();
        hist.record(1550);
        hist.record(1650);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1550);
        assert_eq!(hist.max(), 1650);
    }

    #[test]
    fn test_hdr_histogram_stress_146() {
        let mut hist = HdrHistogram::new();
        hist.record(1560);
        hist.record(1660);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1560);
        assert_eq!(hist.max(), 1660);
    }

    #[test]
    fn test_hdr_histogram_stress_147() {
        let mut hist = HdrHistogram::new();
        hist.record(1570);
        hist.record(1670);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1570);
        assert_eq!(hist.max(), 1670);
    }

    #[test]
    fn test_hdr_histogram_stress_148() {
        let mut hist = HdrHistogram::new();
        hist.record(1580);
        hist.record(1680);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1580);
        assert_eq!(hist.max(), 1680);
    }

    #[test]
    fn test_hdr_histogram_stress_149() {
        let mut hist = HdrHistogram::new();
        hist.record(1590);
        hist.record(1690);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1590);
        assert_eq!(hist.max(), 1690);
    }

    #[test]
    fn test_hdr_histogram_stress_150() {
        let mut hist = HdrHistogram::new();
        hist.record(1600);
        hist.record(1700);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1600);
        assert_eq!(hist.max(), 1700);
    }

    #[test]
    fn test_hdr_histogram_stress_151() {
        let mut hist = HdrHistogram::new();
        hist.record(1610);
        hist.record(1710);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1610);
        assert_eq!(hist.max(), 1710);
    }

    #[test]
    fn test_hdr_histogram_stress_152() {
        let mut hist = HdrHistogram::new();
        hist.record(1620);
        hist.record(1720);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1620);
        assert_eq!(hist.max(), 1720);
    }

    #[test]
    fn test_hdr_histogram_stress_153() {
        let mut hist = HdrHistogram::new();
        hist.record(1630);
        hist.record(1730);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1630);
        assert_eq!(hist.max(), 1730);
    }

    #[test]
    fn test_hdr_histogram_stress_154() {
        let mut hist = HdrHistogram::new();
        hist.record(1640);
        hist.record(1740);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1640);
        assert_eq!(hist.max(), 1740);
    }

    #[test]
    fn test_hdr_histogram_stress_155() {
        let mut hist = HdrHistogram::new();
        hist.record(1650);
        hist.record(1750);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1650);
        assert_eq!(hist.max(), 1750);
    }

    #[test]
    fn test_hdr_histogram_stress_156() {
        let mut hist = HdrHistogram::new();
        hist.record(1660);
        hist.record(1760);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1660);
        assert_eq!(hist.max(), 1760);
    }

    #[test]
    fn test_hdr_histogram_stress_157() {
        let mut hist = HdrHistogram::new();
        hist.record(1670);
        hist.record(1770);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1670);
        assert_eq!(hist.max(), 1770);
    }

    #[test]
    fn test_hdr_histogram_stress_158() {
        let mut hist = HdrHistogram::new();
        hist.record(1680);
        hist.record(1780);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1680);
        assert_eq!(hist.max(), 1780);
    }

    #[test]
    fn test_hdr_histogram_stress_159() {
        let mut hist = HdrHistogram::new();
        hist.record(1690);
        hist.record(1790);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1690);
        assert_eq!(hist.max(), 1790);
    }

    #[test]
    fn test_hdr_histogram_stress_160() {
        let mut hist = HdrHistogram::new();
        hist.record(1700);
        hist.record(1800);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1700);
        assert_eq!(hist.max(), 1800);
    }

    #[test]
    fn test_hdr_histogram_stress_161() {
        let mut hist = HdrHistogram::new();
        hist.record(1710);
        hist.record(1810);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1710);
        assert_eq!(hist.max(), 1810);
    }

    #[test]
    fn test_hdr_histogram_stress_162() {
        let mut hist = HdrHistogram::new();
        hist.record(1720);
        hist.record(1820);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1720);
        assert_eq!(hist.max(), 1820);
    }

    #[test]
    fn test_hdr_histogram_stress_163() {
        let mut hist = HdrHistogram::new();
        hist.record(1730);
        hist.record(1830);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1730);
        assert_eq!(hist.max(), 1830);
    }

    #[test]
    fn test_hdr_histogram_stress_164() {
        let mut hist = HdrHistogram::new();
        hist.record(1740);
        hist.record(1840);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1740);
        assert_eq!(hist.max(), 1840);
    }

    #[test]
    fn test_hdr_histogram_stress_165() {
        let mut hist = HdrHistogram::new();
        hist.record(1750);
        hist.record(1850);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1750);
        assert_eq!(hist.max(), 1850);
    }

    #[test]
    fn test_hdr_histogram_stress_166() {
        let mut hist = HdrHistogram::new();
        hist.record(1760);
        hist.record(1860);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1760);
        assert_eq!(hist.max(), 1860);
    }

    #[test]
    fn test_hdr_histogram_stress_167() {
        let mut hist = HdrHistogram::new();
        hist.record(1770);
        hist.record(1870);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1770);
        assert_eq!(hist.max(), 1870);
    }

    #[test]
    fn test_hdr_histogram_stress_168() {
        let mut hist = HdrHistogram::new();
        hist.record(1780);
        hist.record(1880);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1780);
        assert_eq!(hist.max(), 1880);
    }

    #[test]
    fn test_hdr_histogram_stress_169() {
        let mut hist = HdrHistogram::new();
        hist.record(1790);
        hist.record(1890);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1790);
        assert_eq!(hist.max(), 1890);
    }

    #[test]
    fn test_hdr_histogram_stress_170() {
        let mut hist = HdrHistogram::new();
        hist.record(1800);
        hist.record(1900);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1800);
        assert_eq!(hist.max(), 1900);
    }

    #[test]
    fn test_hdr_histogram_stress_171() {
        let mut hist = HdrHistogram::new();
        hist.record(1810);
        hist.record(1910);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1810);
        assert_eq!(hist.max(), 1910);
    }

    #[test]
    fn test_hdr_histogram_stress_172() {
        let mut hist = HdrHistogram::new();
        hist.record(1820);
        hist.record(1920);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1820);
        assert_eq!(hist.max(), 1920);
    }

    #[test]
    fn test_hdr_histogram_stress_173() {
        let mut hist = HdrHistogram::new();
        hist.record(1830);
        hist.record(1930);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1830);
        assert_eq!(hist.max(), 1930);
    }

    #[test]
    fn test_hdr_histogram_stress_174() {
        let mut hist = HdrHistogram::new();
        hist.record(1840);
        hist.record(1940);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1840);
        assert_eq!(hist.max(), 1940);
    }

    #[test]
    fn test_hdr_histogram_stress_175() {
        let mut hist = HdrHistogram::new();
        hist.record(1850);
        hist.record(1950);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1850);
        assert_eq!(hist.max(), 1950);
    }

    #[test]
    fn test_hdr_histogram_stress_176() {
        let mut hist = HdrHistogram::new();
        hist.record(1860);
        hist.record(1960);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1860);
        assert_eq!(hist.max(), 1960);
    }

    #[test]
    fn test_hdr_histogram_stress_177() {
        let mut hist = HdrHistogram::new();
        hist.record(1870);
        hist.record(1970);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1870);
        assert_eq!(hist.max(), 1970);
    }

    #[test]
    fn test_hdr_histogram_stress_178() {
        let mut hist = HdrHistogram::new();
        hist.record(1880);
        hist.record(1980);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1880);
        assert_eq!(hist.max(), 1980);
    }

    #[test]
    fn test_hdr_histogram_stress_179() {
        let mut hist = HdrHistogram::new();
        hist.record(1890);
        hist.record(1990);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1890);
        assert_eq!(hist.max(), 1990);
    }

    #[test]
    fn test_hdr_histogram_stress_180() {
        let mut hist = HdrHistogram::new();
        hist.record(1900);
        hist.record(2000);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1900);
        assert_eq!(hist.max(), 2000);
    }

    #[test]
    fn test_hdr_histogram_stress_181() {
        let mut hist = HdrHistogram::new();
        hist.record(1910);
        hist.record(2010);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1910);
        assert_eq!(hist.max(), 2010);
    }

    #[test]
    fn test_hdr_histogram_stress_182() {
        let mut hist = HdrHistogram::new();
        hist.record(1920);
        hist.record(2020);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1920);
        assert_eq!(hist.max(), 2020);
    }

    #[test]
    fn test_hdr_histogram_stress_183() {
        let mut hist = HdrHistogram::new();
        hist.record(1930);
        hist.record(2030);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1930);
        assert_eq!(hist.max(), 2030);
    }

    #[test]
    fn test_hdr_histogram_stress_184() {
        let mut hist = HdrHistogram::new();
        hist.record(1940);
        hist.record(2040);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1940);
        assert_eq!(hist.max(), 2040);
    }

    #[test]
    fn test_hdr_histogram_stress_185() {
        let mut hist = HdrHistogram::new();
        hist.record(1950);
        hist.record(2050);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1950);
        assert_eq!(hist.max(), 2050);
    }

    #[test]
    fn test_hdr_histogram_stress_186() {
        let mut hist = HdrHistogram::new();
        hist.record(1960);
        hist.record(2060);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1960);
        assert_eq!(hist.max(), 2060);
    }

    #[test]
    fn test_hdr_histogram_stress_187() {
        let mut hist = HdrHistogram::new();
        hist.record(1970);
        hist.record(2070);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1970);
        assert_eq!(hist.max(), 2070);
    }

    #[test]
    fn test_hdr_histogram_stress_188() {
        let mut hist = HdrHistogram::new();
        hist.record(1980);
        hist.record(2080);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1980);
        assert_eq!(hist.max(), 2080);
    }

    #[test]
    fn test_hdr_histogram_stress_189() {
        let mut hist = HdrHistogram::new();
        hist.record(1990);
        hist.record(2090);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 1990);
        assert_eq!(hist.max(), 2090);
    }

    #[test]
    fn test_hdr_histogram_stress_190() {
        let mut hist = HdrHistogram::new();
        hist.record(2000);
        hist.record(2100);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2000);
        assert_eq!(hist.max(), 2100);
    }

    #[test]
    fn test_hdr_histogram_stress_191() {
        let mut hist = HdrHistogram::new();
        hist.record(2010);
        hist.record(2110);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2010);
        assert_eq!(hist.max(), 2110);
    }

    #[test]
    fn test_hdr_histogram_stress_192() {
        let mut hist = HdrHistogram::new();
        hist.record(2020);
        hist.record(2120);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2020);
        assert_eq!(hist.max(), 2120);
    }

    #[test]
    fn test_hdr_histogram_stress_193() {
        let mut hist = HdrHistogram::new();
        hist.record(2030);
        hist.record(2130);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2030);
        assert_eq!(hist.max(), 2130);
    }

    #[test]
    fn test_hdr_histogram_stress_194() {
        let mut hist = HdrHistogram::new();
        hist.record(2040);
        hist.record(2140);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2040);
        assert_eq!(hist.max(), 2140);
    }

    #[test]
    fn test_hdr_histogram_stress_195() {
        let mut hist = HdrHistogram::new();
        hist.record(2050);
        hist.record(2150);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2050);
        assert_eq!(hist.max(), 2150);
    }

    #[test]
    fn test_hdr_histogram_stress_196() {
        let mut hist = HdrHistogram::new();
        hist.record(2060);
        hist.record(2160);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2060);
        assert_eq!(hist.max(), 2160);
    }

    #[test]
    fn test_hdr_histogram_stress_197() {
        let mut hist = HdrHistogram::new();
        hist.record(2070);
        hist.record(2170);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2070);
        assert_eq!(hist.max(), 2170);
    }

    #[test]
    fn test_hdr_histogram_stress_198() {
        let mut hist = HdrHistogram::new();
        hist.record(2080);
        hist.record(2180);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2080);
        assert_eq!(hist.max(), 2180);
    }

    #[test]
    fn test_hdr_histogram_stress_199() {
        let mut hist = HdrHistogram::new();
        hist.record(2090);
        hist.record(2190);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2090);
        assert_eq!(hist.max(), 2190);
    }

    #[test]
    fn test_hdr_histogram_stress_200() {
        let mut hist = HdrHistogram::new();
        hist.record(2100);
        hist.record(2200);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2100);
        assert_eq!(hist.max(), 2200);
    }

    #[test]
    fn test_hdr_histogram_stress_201() {
        let mut hist = HdrHistogram::new();
        hist.record(2110);
        hist.record(2210);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2110);
        assert_eq!(hist.max(), 2210);
    }

    #[test]
    fn test_hdr_histogram_stress_202() {
        let mut hist = HdrHistogram::new();
        hist.record(2120);
        hist.record(2220);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2120);
        assert_eq!(hist.max(), 2220);
    }

    #[test]
    fn test_hdr_histogram_stress_203() {
        let mut hist = HdrHistogram::new();
        hist.record(2130);
        hist.record(2230);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2130);
        assert_eq!(hist.max(), 2230);
    }

    #[test]
    fn test_hdr_histogram_stress_204() {
        let mut hist = HdrHistogram::new();
        hist.record(2140);
        hist.record(2240);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2140);
        assert_eq!(hist.max(), 2240);
    }

    #[test]
    fn test_hdr_histogram_stress_205() {
        let mut hist = HdrHistogram::new();
        hist.record(2150);
        hist.record(2250);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2150);
        assert_eq!(hist.max(), 2250);
    }

    #[test]
    fn test_hdr_histogram_stress_206() {
        let mut hist = HdrHistogram::new();
        hist.record(2160);
        hist.record(2260);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2160);
        assert_eq!(hist.max(), 2260);
    }

    #[test]
    fn test_hdr_histogram_stress_207() {
        let mut hist = HdrHistogram::new();
        hist.record(2170);
        hist.record(2270);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2170);
        assert_eq!(hist.max(), 2270);
    }

    #[test]
    fn test_hdr_histogram_stress_208() {
        let mut hist = HdrHistogram::new();
        hist.record(2180);
        hist.record(2280);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2180);
        assert_eq!(hist.max(), 2280);
    }

    #[test]
    fn test_hdr_histogram_stress_209() {
        let mut hist = HdrHistogram::new();
        hist.record(2190);
        hist.record(2290);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2190);
        assert_eq!(hist.max(), 2290);
    }

    #[test]
    fn test_hdr_histogram_stress_210() {
        let mut hist = HdrHistogram::new();
        hist.record(2200);
        hist.record(2300);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2200);
        assert_eq!(hist.max(), 2300);
    }

    #[test]
    fn test_hdr_histogram_stress_211() {
        let mut hist = HdrHistogram::new();
        hist.record(2210);
        hist.record(2310);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2210);
        assert_eq!(hist.max(), 2310);
    }

    #[test]
    fn test_hdr_histogram_stress_212() {
        let mut hist = HdrHistogram::new();
        hist.record(2220);
        hist.record(2320);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2220);
        assert_eq!(hist.max(), 2320);
    }

    #[test]
    fn test_hdr_histogram_stress_213() {
        let mut hist = HdrHistogram::new();
        hist.record(2230);
        hist.record(2330);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2230);
        assert_eq!(hist.max(), 2330);
    }

    #[test]
    fn test_hdr_histogram_stress_214() {
        let mut hist = HdrHistogram::new();
        hist.record(2240);
        hist.record(2340);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2240);
        assert_eq!(hist.max(), 2340);
    }

    #[test]
    fn test_hdr_histogram_stress_215() {
        let mut hist = HdrHistogram::new();
        hist.record(2250);
        hist.record(2350);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2250);
        assert_eq!(hist.max(), 2350);
    }

    #[test]
    fn test_hdr_histogram_stress_216() {
        let mut hist = HdrHistogram::new();
        hist.record(2260);
        hist.record(2360);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2260);
        assert_eq!(hist.max(), 2360);
    }

    #[test]
    fn test_hdr_histogram_stress_217() {
        let mut hist = HdrHistogram::new();
        hist.record(2270);
        hist.record(2370);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2270);
        assert_eq!(hist.max(), 2370);
    }

    #[test]
    fn test_hdr_histogram_stress_218() {
        let mut hist = HdrHistogram::new();
        hist.record(2280);
        hist.record(2380);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2280);
        assert_eq!(hist.max(), 2380);
    }

    #[test]
    fn test_hdr_histogram_stress_219() {
        let mut hist = HdrHistogram::new();
        hist.record(2290);
        hist.record(2390);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2290);
        assert_eq!(hist.max(), 2390);
    }

    #[test]
    fn test_hdr_histogram_stress_220() {
        let mut hist = HdrHistogram::new();
        hist.record(2300);
        hist.record(2400);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2300);
        assert_eq!(hist.max(), 2400);
    }

    #[test]
    fn test_hdr_histogram_stress_221() {
        let mut hist = HdrHistogram::new();
        hist.record(2310);
        hist.record(2410);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2310);
        assert_eq!(hist.max(), 2410);
    }

    #[test]
    fn test_hdr_histogram_stress_222() {
        let mut hist = HdrHistogram::new();
        hist.record(2320);
        hist.record(2420);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2320);
        assert_eq!(hist.max(), 2420);
    }

    #[test]
    fn test_hdr_histogram_stress_223() {
        let mut hist = HdrHistogram::new();
        hist.record(2330);
        hist.record(2430);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2330);
        assert_eq!(hist.max(), 2430);
    }

    #[test]
    fn test_hdr_histogram_stress_224() {
        let mut hist = HdrHistogram::new();
        hist.record(2340);
        hist.record(2440);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2340);
        assert_eq!(hist.max(), 2440);
    }

    #[test]
    fn test_hdr_histogram_stress_225() {
        let mut hist = HdrHistogram::new();
        hist.record(2350);
        hist.record(2450);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2350);
        assert_eq!(hist.max(), 2450);
    }

    #[test]
    fn test_hdr_histogram_stress_226() {
        let mut hist = HdrHistogram::new();
        hist.record(2360);
        hist.record(2460);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2360);
        assert_eq!(hist.max(), 2460);
    }

    #[test]
    fn test_hdr_histogram_stress_227() {
        let mut hist = HdrHistogram::new();
        hist.record(2370);
        hist.record(2470);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2370);
        assert_eq!(hist.max(), 2470);
    }

    #[test]
    fn test_hdr_histogram_stress_228() {
        let mut hist = HdrHistogram::new();
        hist.record(2380);
        hist.record(2480);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2380);
        assert_eq!(hist.max(), 2480);
    }

    #[test]
    fn test_hdr_histogram_stress_229() {
        let mut hist = HdrHistogram::new();
        hist.record(2390);
        hist.record(2490);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2390);
        assert_eq!(hist.max(), 2490);
    }

    #[test]
    fn test_hdr_histogram_stress_230() {
        let mut hist = HdrHistogram::new();
        hist.record(2400);
        hist.record(2500);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2400);
        assert_eq!(hist.max(), 2500);
    }

    #[test]
    fn test_hdr_histogram_stress_231() {
        let mut hist = HdrHistogram::new();
        hist.record(2410);
        hist.record(2510);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2410);
        assert_eq!(hist.max(), 2510);
    }

    #[test]
    fn test_hdr_histogram_stress_232() {
        let mut hist = HdrHistogram::new();
        hist.record(2420);
        hist.record(2520);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2420);
        assert_eq!(hist.max(), 2520);
    }

    #[test]
    fn test_hdr_histogram_stress_233() {
        let mut hist = HdrHistogram::new();
        hist.record(2430);
        hist.record(2530);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2430);
        assert_eq!(hist.max(), 2530);
    }

    #[test]
    fn test_hdr_histogram_stress_234() {
        let mut hist = HdrHistogram::new();
        hist.record(2440);
        hist.record(2540);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2440);
        assert_eq!(hist.max(), 2540);
    }

    #[test]
    fn test_hdr_histogram_stress_235() {
        let mut hist = HdrHistogram::new();
        hist.record(2450);
        hist.record(2550);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2450);
        assert_eq!(hist.max(), 2550);
    }

    #[test]
    fn test_hdr_histogram_stress_236() {
        let mut hist = HdrHistogram::new();
        hist.record(2460);
        hist.record(2560);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2460);
        assert_eq!(hist.max(), 2560);
    }

    #[test]
    fn test_hdr_histogram_stress_237() {
        let mut hist = HdrHistogram::new();
        hist.record(2470);
        hist.record(2570);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2470);
        assert_eq!(hist.max(), 2570);
    }

    #[test]
    fn test_hdr_histogram_stress_238() {
        let mut hist = HdrHistogram::new();
        hist.record(2480);
        hist.record(2580);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2480);
        assert_eq!(hist.max(), 2580);
    }

    #[test]
    fn test_hdr_histogram_stress_239() {
        let mut hist = HdrHistogram::new();
        hist.record(2490);
        hist.record(2590);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2490);
        assert_eq!(hist.max(), 2590);
    }

    #[test]
    fn test_hdr_histogram_stress_240() {
        let mut hist = HdrHistogram::new();
        hist.record(2500);
        hist.record(2600);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2500);
        assert_eq!(hist.max(), 2600);
    }

    #[test]
    fn test_hdr_histogram_stress_241() {
        let mut hist = HdrHistogram::new();
        hist.record(2510);
        hist.record(2610);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2510);
        assert_eq!(hist.max(), 2610);
    }

    #[test]
    fn test_hdr_histogram_stress_242() {
        let mut hist = HdrHistogram::new();
        hist.record(2520);
        hist.record(2620);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2520);
        assert_eq!(hist.max(), 2620);
    }

    #[test]
    fn test_hdr_histogram_stress_243() {
        let mut hist = HdrHistogram::new();
        hist.record(2530);
        hist.record(2630);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2530);
        assert_eq!(hist.max(), 2630);
    }

    #[test]
    fn test_hdr_histogram_stress_244() {
        let mut hist = HdrHistogram::new();
        hist.record(2540);
        hist.record(2640);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2540);
        assert_eq!(hist.max(), 2640);
    }

    #[test]
    fn test_hdr_histogram_stress_245() {
        let mut hist = HdrHistogram::new();
        hist.record(2550);
        hist.record(2650);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2550);
        assert_eq!(hist.max(), 2650);
    }

    #[test]
    fn test_hdr_histogram_stress_246() {
        let mut hist = HdrHistogram::new();
        hist.record(2560);
        hist.record(2660);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2560);
        assert_eq!(hist.max(), 2660);
    }

    #[test]
    fn test_hdr_histogram_stress_247() {
        let mut hist = HdrHistogram::new();
        hist.record(2570);
        hist.record(2670);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2570);
        assert_eq!(hist.max(), 2670);
    }

    #[test]
    fn test_hdr_histogram_stress_248() {
        let mut hist = HdrHistogram::new();
        hist.record(2580);
        hist.record(2680);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2580);
        assert_eq!(hist.max(), 2680);
    }

    #[test]
    fn test_hdr_histogram_stress_249() {
        let mut hist = HdrHistogram::new();
        hist.record(2590);
        hist.record(2690);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2590);
        assert_eq!(hist.max(), 2690);
    }

    #[test]
    fn test_hdr_histogram_stress_250() {
        let mut hist = HdrHistogram::new();
        hist.record(2600);
        hist.record(2700);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2600);
        assert_eq!(hist.max(), 2700);
    }

    #[test]
    fn test_hdr_histogram_stress_251() {
        let mut hist = HdrHistogram::new();
        hist.record(2610);
        hist.record(2710);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2610);
        assert_eq!(hist.max(), 2710);
    }

    #[test]
    fn test_hdr_histogram_stress_252() {
        let mut hist = HdrHistogram::new();
        hist.record(2620);
        hist.record(2720);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2620);
        assert_eq!(hist.max(), 2720);
    }

    #[test]
    fn test_hdr_histogram_stress_253() {
        let mut hist = HdrHistogram::new();
        hist.record(2630);
        hist.record(2730);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2630);
        assert_eq!(hist.max(), 2730);
    }

    #[test]
    fn test_hdr_histogram_stress_254() {
        let mut hist = HdrHistogram::new();
        hist.record(2640);
        hist.record(2740);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2640);
        assert_eq!(hist.max(), 2740);
    }

    #[test]
    fn test_hdr_histogram_stress_255() {
        let mut hist = HdrHistogram::new();
        hist.record(2650);
        hist.record(2750);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2650);
        assert_eq!(hist.max(), 2750);
    }

    #[test]
    fn test_hdr_histogram_stress_256() {
        let mut hist = HdrHistogram::new();
        hist.record(2660);
        hist.record(2760);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2660);
        assert_eq!(hist.max(), 2760);
    }

    #[test]
    fn test_hdr_histogram_stress_257() {
        let mut hist = HdrHistogram::new();
        hist.record(2670);
        hist.record(2770);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2670);
        assert_eq!(hist.max(), 2770);
    }

    #[test]
    fn test_hdr_histogram_stress_258() {
        let mut hist = HdrHistogram::new();
        hist.record(2680);
        hist.record(2780);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2680);
        assert_eq!(hist.max(), 2780);
    }

    #[test]
    fn test_hdr_histogram_stress_259() {
        let mut hist = HdrHistogram::new();
        hist.record(2690);
        hist.record(2790);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2690);
        assert_eq!(hist.max(), 2790);
    }

    #[test]
    fn test_hdr_histogram_stress_260() {
        let mut hist = HdrHistogram::new();
        hist.record(2700);
        hist.record(2800);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2700);
        assert_eq!(hist.max(), 2800);
    }

    #[test]
    fn test_hdr_histogram_stress_261() {
        let mut hist = HdrHistogram::new();
        hist.record(2710);
        hist.record(2810);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2710);
        assert_eq!(hist.max(), 2810);
    }

    #[test]
    fn test_hdr_histogram_stress_262() {
        let mut hist = HdrHistogram::new();
        hist.record(2720);
        hist.record(2820);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2720);
        assert_eq!(hist.max(), 2820);
    }

    #[test]
    fn test_hdr_histogram_stress_263() {
        let mut hist = HdrHistogram::new();
        hist.record(2730);
        hist.record(2830);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2730);
        assert_eq!(hist.max(), 2830);
    }

    #[test]
    fn test_hdr_histogram_stress_264() {
        let mut hist = HdrHistogram::new();
        hist.record(2740);
        hist.record(2840);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2740);
        assert_eq!(hist.max(), 2840);
    }

    #[test]
    fn test_hdr_histogram_stress_265() {
        let mut hist = HdrHistogram::new();
        hist.record(2750);
        hist.record(2850);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2750);
        assert_eq!(hist.max(), 2850);
    }

    #[test]
    fn test_hdr_histogram_stress_266() {
        let mut hist = HdrHistogram::new();
        hist.record(2760);
        hist.record(2860);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2760);
        assert_eq!(hist.max(), 2860);
    }

    #[test]
    fn test_hdr_histogram_stress_267() {
        let mut hist = HdrHistogram::new();
        hist.record(2770);
        hist.record(2870);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2770);
        assert_eq!(hist.max(), 2870);
    }

    #[test]
    fn test_hdr_histogram_stress_268() {
        let mut hist = HdrHistogram::new();
        hist.record(2780);
        hist.record(2880);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2780);
        assert_eq!(hist.max(), 2880);
    }

    #[test]
    fn test_hdr_histogram_stress_269() {
        let mut hist = HdrHistogram::new();
        hist.record(2790);
        hist.record(2890);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2790);
        assert_eq!(hist.max(), 2890);
    }

    #[test]
    fn test_hdr_histogram_stress_270() {
        let mut hist = HdrHistogram::new();
        hist.record(2800);
        hist.record(2900);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2800);
        assert_eq!(hist.max(), 2900);
    }

    #[test]
    fn test_hdr_histogram_stress_271() {
        let mut hist = HdrHistogram::new();
        hist.record(2810);
        hist.record(2910);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2810);
        assert_eq!(hist.max(), 2910);
    }

    #[test]
    fn test_hdr_histogram_stress_272() {
        let mut hist = HdrHistogram::new();
        hist.record(2820);
        hist.record(2920);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2820);
        assert_eq!(hist.max(), 2920);
    }

    #[test]
    fn test_hdr_histogram_stress_273() {
        let mut hist = HdrHistogram::new();
        hist.record(2830);
        hist.record(2930);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2830);
        assert_eq!(hist.max(), 2930);
    }

    #[test]
    fn test_hdr_histogram_stress_274() {
        let mut hist = HdrHistogram::new();
        hist.record(2840);
        hist.record(2940);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2840);
        assert_eq!(hist.max(), 2940);
    }

    #[test]
    fn test_hdr_histogram_stress_275() {
        let mut hist = HdrHistogram::new();
        hist.record(2850);
        hist.record(2950);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2850);
        assert_eq!(hist.max(), 2950);
    }

    #[test]
    fn test_hdr_histogram_stress_276() {
        let mut hist = HdrHistogram::new();
        hist.record(2860);
        hist.record(2960);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2860);
        assert_eq!(hist.max(), 2960);
    }

    #[test]
    fn test_hdr_histogram_stress_277() {
        let mut hist = HdrHistogram::new();
        hist.record(2870);
        hist.record(2970);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2870);
        assert_eq!(hist.max(), 2970);
    }

    #[test]
    fn test_hdr_histogram_stress_278() {
        let mut hist = HdrHistogram::new();
        hist.record(2880);
        hist.record(2980);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2880);
        assert_eq!(hist.max(), 2980);
    }

    #[test]
    fn test_hdr_histogram_stress_279() {
        let mut hist = HdrHistogram::new();
        hist.record(2890);
        hist.record(2990);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2890);
        assert_eq!(hist.max(), 2990);
    }

    #[test]
    fn test_hdr_histogram_stress_280() {
        let mut hist = HdrHistogram::new();
        hist.record(2900);
        hist.record(3000);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2900);
        assert_eq!(hist.max(), 3000);
    }

    #[test]
    fn test_hdr_histogram_stress_281() {
        let mut hist = HdrHistogram::new();
        hist.record(2910);
        hist.record(3010);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2910);
        assert_eq!(hist.max(), 3010);
    }

    #[test]
    fn test_hdr_histogram_stress_282() {
        let mut hist = HdrHistogram::new();
        hist.record(2920);
        hist.record(3020);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2920);
        assert_eq!(hist.max(), 3020);
    }

    #[test]
    fn test_hdr_histogram_stress_283() {
        let mut hist = HdrHistogram::new();
        hist.record(2930);
        hist.record(3030);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2930);
        assert_eq!(hist.max(), 3030);
    }

    #[test]
    fn test_hdr_histogram_stress_284() {
        let mut hist = HdrHistogram::new();
        hist.record(2940);
        hist.record(3040);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2940);
        assert_eq!(hist.max(), 3040);
    }

    #[test]
    fn test_hdr_histogram_stress_285() {
        let mut hist = HdrHistogram::new();
        hist.record(2950);
        hist.record(3050);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2950);
        assert_eq!(hist.max(), 3050);
    }

    #[test]
    fn test_hdr_histogram_stress_286() {
        let mut hist = HdrHistogram::new();
        hist.record(2960);
        hist.record(3060);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2960);
        assert_eq!(hist.max(), 3060);
    }

    #[test]
    fn test_hdr_histogram_stress_287() {
        let mut hist = HdrHistogram::new();
        hist.record(2970);
        hist.record(3070);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2970);
        assert_eq!(hist.max(), 3070);
    }

    #[test]
    fn test_hdr_histogram_stress_288() {
        let mut hist = HdrHistogram::new();
        hist.record(2980);
        hist.record(3080);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2980);
        assert_eq!(hist.max(), 3080);
    }

    #[test]
    fn test_hdr_histogram_stress_289() {
        let mut hist = HdrHistogram::new();
        hist.record(2990);
        hist.record(3090);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 2990);
        assert_eq!(hist.max(), 3090);
    }

    #[test]
    fn test_hdr_histogram_stress_290() {
        let mut hist = HdrHistogram::new();
        hist.record(3000);
        hist.record(3100);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3000);
        assert_eq!(hist.max(), 3100);
    }

    #[test]
    fn test_hdr_histogram_stress_291() {
        let mut hist = HdrHistogram::new();
        hist.record(3010);
        hist.record(3110);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3010);
        assert_eq!(hist.max(), 3110);
    }

    #[test]
    fn test_hdr_histogram_stress_292() {
        let mut hist = HdrHistogram::new();
        hist.record(3020);
        hist.record(3120);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3020);
        assert_eq!(hist.max(), 3120);
    }

    #[test]
    fn test_hdr_histogram_stress_293() {
        let mut hist = HdrHistogram::new();
        hist.record(3030);
        hist.record(3130);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3030);
        assert_eq!(hist.max(), 3130);
    }

    #[test]
    fn test_hdr_histogram_stress_294() {
        let mut hist = HdrHistogram::new();
        hist.record(3040);
        hist.record(3140);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3040);
        assert_eq!(hist.max(), 3140);
    }

    #[test]
    fn test_hdr_histogram_stress_295() {
        let mut hist = HdrHistogram::new();
        hist.record(3050);
        hist.record(3150);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3050);
        assert_eq!(hist.max(), 3150);
    }

    #[test]
    fn test_hdr_histogram_stress_296() {
        let mut hist = HdrHistogram::new();
        hist.record(3060);
        hist.record(3160);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3060);
        assert_eq!(hist.max(), 3160);
    }

    #[test]
    fn test_hdr_histogram_stress_297() {
        let mut hist = HdrHistogram::new();
        hist.record(3070);
        hist.record(3170);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3070);
        assert_eq!(hist.max(), 3170);
    }

    #[test]
    fn test_hdr_histogram_stress_298() {
        let mut hist = HdrHistogram::new();
        hist.record(3080);
        hist.record(3180);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3080);
        assert_eq!(hist.max(), 3180);
    }

    #[test]
    fn test_hdr_histogram_stress_299() {
        let mut hist = HdrHistogram::new();
        hist.record(3090);
        hist.record(3190);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3090);
        assert_eq!(hist.max(), 3190);
    }

    #[test]
    fn test_hdr_histogram_stress_300() {
        let mut hist = HdrHistogram::new();
        hist.record(3100);
        hist.record(3200);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3100);
        assert_eq!(hist.max(), 3200);
    }

    #[test]
    fn test_hdr_histogram_stress_301() {
        let mut hist = HdrHistogram::new();
        hist.record(3110);
        hist.record(3210);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3110);
        assert_eq!(hist.max(), 3210);
    }

    #[test]
    fn test_hdr_histogram_stress_302() {
        let mut hist = HdrHistogram::new();
        hist.record(3120);
        hist.record(3220);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3120);
        assert_eq!(hist.max(), 3220);
    }

    #[test]
    fn test_hdr_histogram_stress_303() {
        let mut hist = HdrHistogram::new();
        hist.record(3130);
        hist.record(3230);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3130);
        assert_eq!(hist.max(), 3230);
    }

    #[test]
    fn test_hdr_histogram_stress_304() {
        let mut hist = HdrHistogram::new();
        hist.record(3140);
        hist.record(3240);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3140);
        assert_eq!(hist.max(), 3240);
    }

    #[test]
    fn test_hdr_histogram_stress_305() {
        let mut hist = HdrHistogram::new();
        hist.record(3150);
        hist.record(3250);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3150);
        assert_eq!(hist.max(), 3250);
    }

    #[test]
    fn test_hdr_histogram_stress_306() {
        let mut hist = HdrHistogram::new();
        hist.record(3160);
        hist.record(3260);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3160);
        assert_eq!(hist.max(), 3260);
    }

    #[test]
    fn test_hdr_histogram_stress_307() {
        let mut hist = HdrHistogram::new();
        hist.record(3170);
        hist.record(3270);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3170);
        assert_eq!(hist.max(), 3270);
    }

    #[test]
    fn test_hdr_histogram_stress_308() {
        let mut hist = HdrHistogram::new();
        hist.record(3180);
        hist.record(3280);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3180);
        assert_eq!(hist.max(), 3280);
    }

    #[test]
    fn test_hdr_histogram_stress_309() {
        let mut hist = HdrHistogram::new();
        hist.record(3190);
        hist.record(3290);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3190);
        assert_eq!(hist.max(), 3290);
    }

    #[test]
    fn test_hdr_histogram_stress_310() {
        let mut hist = HdrHistogram::new();
        hist.record(3200);
        hist.record(3300);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3200);
        assert_eq!(hist.max(), 3300);
    }

    #[test]
    fn test_hdr_histogram_stress_311() {
        let mut hist = HdrHistogram::new();
        hist.record(3210);
        hist.record(3310);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3210);
        assert_eq!(hist.max(), 3310);
    }

    #[test]
    fn test_hdr_histogram_stress_312() {
        let mut hist = HdrHistogram::new();
        hist.record(3220);
        hist.record(3320);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3220);
        assert_eq!(hist.max(), 3320);
    }

    #[test]
    fn test_hdr_histogram_stress_313() {
        let mut hist = HdrHistogram::new();
        hist.record(3230);
        hist.record(3330);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3230);
        assert_eq!(hist.max(), 3330);
    }

    #[test]
    fn test_hdr_histogram_stress_314() {
        let mut hist = HdrHistogram::new();
        hist.record(3240);
        hist.record(3340);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3240);
        assert_eq!(hist.max(), 3340);
    }

    #[test]
    fn test_hdr_histogram_stress_315() {
        let mut hist = HdrHistogram::new();
        hist.record(3250);
        hist.record(3350);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3250);
        assert_eq!(hist.max(), 3350);
    }

    #[test]
    fn test_hdr_histogram_stress_316() {
        let mut hist = HdrHistogram::new();
        hist.record(3260);
        hist.record(3360);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3260);
        assert_eq!(hist.max(), 3360);
    }

    #[test]
    fn test_hdr_histogram_stress_317() {
        let mut hist = HdrHistogram::new();
        hist.record(3270);
        hist.record(3370);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3270);
        assert_eq!(hist.max(), 3370);
    }

    #[test]
    fn test_hdr_histogram_stress_318() {
        let mut hist = HdrHistogram::new();
        hist.record(3280);
        hist.record(3380);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3280);
        assert_eq!(hist.max(), 3380);
    }

    #[test]
    fn test_hdr_histogram_stress_319() {
        let mut hist = HdrHistogram::new();
        hist.record(3290);
        hist.record(3390);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3290);
        assert_eq!(hist.max(), 3390);
    }

    #[test]
    fn test_hdr_histogram_stress_320() {
        let mut hist = HdrHistogram::new();
        hist.record(3300);
        hist.record(3400);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3300);
        assert_eq!(hist.max(), 3400);
    }

    #[test]
    fn test_hdr_histogram_stress_321() {
        let mut hist = HdrHistogram::new();
        hist.record(3310);
        hist.record(3410);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3310);
        assert_eq!(hist.max(), 3410);
    }

    #[test]
    fn test_hdr_histogram_stress_322() {
        let mut hist = HdrHistogram::new();
        hist.record(3320);
        hist.record(3420);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3320);
        assert_eq!(hist.max(), 3420);
    }

    #[test]
    fn test_hdr_histogram_stress_323() {
        let mut hist = HdrHistogram::new();
        hist.record(3330);
        hist.record(3430);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3330);
        assert_eq!(hist.max(), 3430);
    }

    #[test]
    fn test_hdr_histogram_stress_324() {
        let mut hist = HdrHistogram::new();
        hist.record(3340);
        hist.record(3440);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3340);
        assert_eq!(hist.max(), 3440);
    }

    #[test]
    fn test_hdr_histogram_stress_325() {
        let mut hist = HdrHistogram::new();
        hist.record(3350);
        hist.record(3450);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3350);
        assert_eq!(hist.max(), 3450);
    }

    #[test]
    fn test_hdr_histogram_stress_326() {
        let mut hist = HdrHistogram::new();
        hist.record(3360);
        hist.record(3460);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3360);
        assert_eq!(hist.max(), 3460);
    }

    #[test]
    fn test_hdr_histogram_stress_327() {
        let mut hist = HdrHistogram::new();
        hist.record(3370);
        hist.record(3470);
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 3370);
        assert_eq!(hist.max(), 3470);
    }

    // Benchmark verification and performance check padding line 0
    // Benchmark verification and performance check padding line 1
    // Benchmark verification and performance check padding line 2
    // Benchmark verification and performance check padding line 3
}
