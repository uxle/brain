//! # Brain Utils Test Suite
use brain_utils::checksum::{Crc32, Adler32};
use brain_utils::prof::timer::TimingStats;
use brain_utils::hal::safety::{SafetyGuard, SafetyConfig};
use brain_utils::hal::hid::{HidAction, KeyAction};
use std::time::Duration;

#[test]
fn test_crc32_and_adler32_checksums() {
    let data = b"Brain Deep Learning Framework";
    let crc = Crc32::compute(data);
    assert_ne!(crc, 0);

    let adler = Adler32::compute(data);
    assert_ne!(adler, 0);

    // Consistency check
    assert_eq!(Crc32::compute(data), crc);
    assert_eq!(Adler32::compute(data), adler);
}

#[test]
fn test_timing_stats_accumulation() {
    let mut stats = TimingStats::new();
    stats.add_sample(Duration::from_millis(10));
    stats.add_sample(Duration::from_millis(20));
    stats.add_sample(Duration::from_millis(30));

    assert_eq!(stats.count, 3);
    assert_eq!(stats.mean(), Duration::from_millis(20));
    assert_eq!(stats.min, Duration::from_millis(10));
    assert_eq!(stats.max, Duration::from_millis(30));
}

#[test]
fn test_hal_safety_guard_blocks_forbidden_commands() {
    let guard = SafetyGuard::new(SafetyConfig::default());
    let dangerous_action = HidAction::Key(KeyAction::TypeStr("rm -rf /".into()));
    let check = guard.verify_action(&dangerous_action);
    assert!(check.is_err(), "SafetyGuard must block 'rm -rf'");

    let safe_action = HidAction::Key(KeyAction::TypeStr("hello world".into()));
    let check_safe = guard.verify_action(&safe_action);
    assert!(check_safe.is_ok(), "SafetyGuard should allow safe typing");
}
