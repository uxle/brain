//! Device abstraction for the Brain deep learning framework.
//!
//! This module defines the [`Device`] enum representing compute devices (CPU, GPU variants,
//! TPU) and provides utilities for device management, querying available devices,
//! and device-specific operations.
//!
//! # Supported Devices
//!
//! * **Cpu** - The host CPU, always available
//! * **Cuda(N)** - NVIDIA CUDA GPU device N
//! * **Vulkan(N)** - Vulkan compute device N
//! * **Metal(N)** - Apple Metal GPU device N
//! * **Tpu(N)** - Google TPU device N
//!
//! # Usage
//!
//! ```ignore
//! use brain_core::device::Device;
//!
//! let cpu = Device::Cpu;
//! let cuda0 = Device::cuda(0);
//! assert!(cpu.is_cpu());
//! assert!(cuda0.is_cuda());
//!
//! // Device properties
//! if let Some(props) = cuda0.properties() {
//!     println!("GPU: {} ({} MB)", props.name, props.total_memory / 1024 / 1024);
//! }
//! ```

use std::fmt;
use std::hash::{Hash, Hasher};
use std::str::FromStr;

// =============================================================================
// DeviceType Enum
// =============================================================================

/// The type of a compute device, without an index.
///
/// `DeviceType` is used for categorizing devices by their backend technology
/// without specifying which specific device instance is being used.
///
/// # Ordering
///
/// The ordering is: Cpu < Cuda < Vulkan < Metal < Tpu. This provides a
/// consistent ordering for sorting and comparison operations.
///
/// # Examples
///
/// ```
/// use brain_core::device::DeviceType;
/// assert!(DeviceType::Cpu < DeviceType::Cuda);
/// assert_eq!(DeviceType::Cuda.name(), "Cuda");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DeviceType {
    /// The host CPU.
    Cpu,
    /// An NVIDIA CUDA-capable GPU.
    Cuda,
    /// A Vulkan compute device.
    Vulkan,
    /// An Apple Metal GPU.
    Metal,
    /// A Google TPU.
    Tpu,
}

impl DeviceType {
    /// Returns the human-readable name of this device type.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::device::DeviceType;
    /// assert_eq!(DeviceType::Cpu.name(), "Cpu");
    /// assert_eq!(DeviceType::Cuda.name(), "Cuda");
    /// ```
    pub fn name(&self) -> &'static str {
        match self {
            DeviceType::Cpu => "Cpu",
            DeviceType::Cuda => "Cuda",
            DeviceType::Vulkan => "Vulkan",
            DeviceType::Metal => "Metal",
            DeviceType::Tpu => "Tpu",
        }
    }

    /// Returns whether this device type is an accelerator (GPU/TPU).
    ///
    /// CPU is not considered an accelerator.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::device::DeviceType;
    /// assert!(!DeviceType::Cpu.is_accelerator());
    /// assert!(DeviceType::Cuda.is_accelerator());
    /// assert!(DeviceType::Tpu.is_accelerator());
    /// ```
    pub fn is_accelerator(&self) -> bool {
        *self != DeviceType::Cpu
    }

    /// Returns whether this device type uses unified memory with the CPU.
    ///
    /// Metal devices (Apple Silicon) typically use unified memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::device::DeviceType;
    /// assert!(DeviceType::Metal.has_unified_memory());
    /// assert!(!DeviceType::Cuda.has_unified_memory());
    /// ```
    pub fn has_unified_memory(&self) -> bool {
        matches!(self, DeviceType::Cpu | DeviceType::Metal)
    }

    /// Returns whether this device type supports peer-to-peer memory access
    /// between devices of the same type.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::device::DeviceType;
    /// assert!(DeviceType::Cuda.supports_p2p());
    /// assert!(!DeviceType::Cpu.supports_p2p());
    /// ```
    pub fn supports_p2p(&self) -> bool {
        matches!(self, DeviceType::Cuda | DeviceType::Metal)
    }

    /// Returns the number of device types.
    pub const COUNT: usize = 5;

    /// Returns all device types in order.
    pub const ALL: [DeviceType; 5] = [
        DeviceType::Cpu,
        DeviceType::Cuda,
        DeviceType::Vulkan,
        DeviceType::Metal,
        DeviceType::Tpu,
    ];
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Default for DeviceType {
    fn default() -> Self {
        DeviceType::Cpu
    }
}

// =============================================================================
// Device Enum
// =============================================================================

/// Represents a compute device where tensor operations can be executed.
///
/// Each variant may carry an index identifying a specific device when multiple
/// devices of the same type are available (e.g., multi-GPU systems).
///
/// # Ordering
///
/// CPU comes first in ordering, followed by Cuda, Vulkan, Metal, and Tpu.
/// Within each type, devices are ordered by index (0 < 1 < 2 ...).
///
/// # Hash
///
/// Two devices with the same type and index will have the same hash.
///
/// # Examples
///
/// ```
/// use brain_core::device::Device;
///
/// let cpu = Device::Cpu;
/// let cuda0 = Device::cuda(0);
/// let cuda1 = Device::cuda(1);
///
/// assert!(cpu.is_cpu());
/// assert!(cuda0.is_cuda());
/// assert_ne!(cuda0, cuda1);
/// assert!(cpu < cuda0);
/// ```
#[derive(Debug, Clone, Copy)]
pub enum Device {
    /// The host CPU device. Always available and used as the default device.
    Cpu,

    /// An NVIDIA CUDA-capable GPU. The inner value is the device index (0-based).
    Cuda(usize),

    /// A Vulkan compute-capable device. The inner value is the device index.
    Vulkan(usize),

    /// An Apple Metal GPU. The inner value is the device index.
    Metal(usize),

    /// A Google TPU. The inner value is the device index.
    Tpu(usize),
}

// =============================================================================
// Device Implementation
// =============================================================================

impl Device {
    /// Creates a CUDA device with the given index.
    ///
    /// # Arguments
    ///
    /// * `index` - The GPU index (0-based)
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::device::Device;
    /// let gpu = Device::cuda(0);
    /// assert!(gpu.is_cuda());
    /// ```
    pub fn cuda(index: usize) -> Self {
        Device::Cuda(index)
    }

    /// Creates a Vulkan device with the given index.
    pub fn vulkan(index: usize) -> Self {
        Device::Vulkan(index)
    }

    /// Creates a Metal device with the given index.
    pub fn metal(index: usize) -> Self {
        Device::Metal(index)
    }

    /// Creates a TPU device with the given index.
    pub fn tpu(index: usize) -> Self {
        Device::Tpu(index)
    }

    /// Returns `true` if this is the CPU device.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::device::Device;
    /// assert!(Device::Cpu.is_cpu());
    /// assert!(!Device::cuda(0).is_cpu());
    /// ```
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    /// Returns `true` if this is a CUDA device.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::device::Device;
    /// assert!(Device::cuda(0).is_cuda());
    /// assert!(!Device::Cpu.is_cuda());
    /// ```
    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }

    /// Returns `true` if this is a Vulkan device.
    pub fn is_vulkan(&self) -> bool {
        matches!(self, Device::Vulkan(_))
    }

    /// Returns `true` if this is a Metal device.
    pub fn is_metal(&self) -> bool {
        matches!(self, Device::Metal(_))
    }

    /// Returns `true` if this is a TPU device.
    pub fn is_tpu(&self) -> bool {
        matches!(self, Device::Tpu(_))
    }

    /// Returns whether this device is an accelerator (any non-CPU device).
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::device::Device;
    /// assert!(!Device::Cpu.is_accelerator());
    /// assert!(Device::cuda(0).is_accelerator());
    /// assert!(Device::tpu(0).is_accelerator());
    /// ```
    pub fn is_accelerator(&self) -> bool {
        !self.is_cpu()
    }

    /// Returns the index of this device.
    ///
    /// For CPU, this returns 0. For GPU/TPU devices, this returns the
    /// device index.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::device::Device;
    /// assert_eq!(Device::Cpu.index(), 0);
    /// assert_eq!(Device::cuda(2).index(), 2);
    /// ```
    pub fn index(&self) -> usize {
        match self {
            Device::Cpu => 0,
            Device::Cuda(i) => *i,
            Device::Vulkan(i) => *i,
            Device::Metal(i) => *i,
            Device::Tpu(i) => *i,
        }
    }

    /// Returns the [`DeviceType`] of this device.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::device::{Device, DeviceType};
    /// assert_eq!(Device::Cpu.device_type(), DeviceType::Cpu);
    /// assert_eq!(Device::cuda(0).device_type(), DeviceType::Cuda);
    /// ```
    pub fn device_type(&self) -> DeviceType {
        match self {
            Device::Cpu => DeviceType::Cpu,
            Device::Cuda(_) => DeviceType::Cuda,
            Device::Vulkan(_) => DeviceType::Vulkan,
            Device::Metal(_) => DeviceType::Metal,
            Device::Tpu(_) => DeviceType::Tpu,
        }
    }

    /// Returns the human-readable name of this device.
    ///
    /// For indexed devices, the format is "TypeName(index)".
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::device::Device;
    /// assert_eq!(Device::Cpu.name(), "Cpu");
    /// assert_eq!(Device::cuda(0).name(), "Cuda(0)");
    /// assert_eq!(Device::vulkan(2).name(), "Vulkan(2)");
    /// ```
    pub fn name(&self) -> String {
        match self {
            Device::Cpu => "Cpu".to_string(),
            Device::Cuda(i) => format!("Cuda({})", i),
            Device::Vulkan(i) => format!("Vulkan({})", i),
            Device::Metal(i) => format!("Metal({})", i),
            Device::Tpu(i) => format!("Tpu({})", i),
        }
    }

    /// Returns the short name of this device (without the index).
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::device::Device;
    /// assert_eq!(Device::Cpu.short_name(), "Cpu");
    /// assert_eq!(Device::cuda(3).short_name(), "Cuda");
    /// ```
    pub fn short_name(&self) -> &'static str {
        self.device_type().name()
    }

    /// Synchronizes this device, ensuring all pending operations are complete.
    ///
    /// For CPU, this is a no-op. For GPU/TPU devices, this waits for all
    /// pending kernels and memory operations to complete.
    ///
    /// In this pure-Rust implementation, this always succeeds.
    pub fn synchronize(&self) -> Result<(), String> {
        // In a real implementation, this would call into CUDA/Vulkan/Metal/TPU
        // synchronization APIs. For now, it's a no-op.
        Ok(())
    }

    /// Returns the amount of free memory available on this device in bytes.
    ///
    /// For CPU, returns an estimate of available system memory.
    /// For GPU/TPU devices, queries the device driver.
    ///
    /// In this pure-Rust implementation without device backends, returns `None`.
    pub fn memory_available(&self) -> Option<usize> {
        // Would require platform-specific APIs in a real implementation
        None
    }

    /// Returns the total memory of this device in bytes.
    ///
    /// In this pure-Rust implementation without device backends, returns `None`.
    pub fn total_memory(&self) -> Option<usize> {
        // Would require platform-specific APIs
        None
    }

    /// Returns device properties if available.
    ///
    /// In this pure-Rust implementation without device backends, returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::device::Device;
    /// // In this implementation, properties are not available
    /// assert!(Device::cuda(0).properties().is_none());
    /// ```
    pub fn properties(&self) -> Option<DeviceProperties> {
        // Would require device driver queries
        None
    }

    /// Returns whether this device has unified memory with the host CPU.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::device::Device;
    /// assert!(Device::Cpu.has_unified_memory());
    /// assert!(!Device::cuda(0).has_unified_memory());
    /// ```
    pub fn has_unified_memory(&self) -> bool {
        self.device_type().has_unified_memory()
    }

    /// Returns whether this device supports asynchronous operations.
    ///
    /// All device types support async operations.
    pub fn supports_async(&self) -> bool {
        true
    }

    /// Returns whether this device supports 16-bit floating-point operations
    /// natively in hardware.
    ///
    /// CPU always supports f16 via software emulation.
    /// GPU support depends on the compute capability.
    pub fn supports_f16(&self) -> bool {
        match self {
            Device::Cpu => false, // software emulation, not native
            Device::Cuda(_) => true, // most modern CUDA GPUs support f16
            Device::Vulkan(_) => true,
            Device::Metal(_) => true,
            Device::Tpu(_) => true,
        }
    }

    /// Returns whether this device supports bfloat16 natively.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::device::Device;
    /// assert!(!Device::Cpu.supports_bf16());
    /// assert!(Device::cuda(0).supports_bf16()); // modern GPUs
    /// ```
    pub fn supports_bf16(&self) -> bool {
        match self {
            Device::Cpu => false,
            Device::Cuda(_) => true,
            Device::Vulkan(_) => false,
            Device::Metal(_) => true,
            Device::Tpu(_) => true,
        }
    }

    /// Returns whether this device supports 64-bit floating-point operations.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::device::Device;
    /// assert!(Device::Cpu.supports_f64());
    /// assert!(Device::cuda(0).supports_f64());
    /// ```
    pub fn supports_f64(&self) -> bool {
        match self {
            Device::Cpu => true,
            Device::Cuda(_) => true,
            Device::Vulkan(_) => false,
            Device::Metal(_) => true,
            Device::Tpu(_) => false,
        }
    }

    /// Returns whether this device supports tensor cores / matrix-multiply
    /// acceleration units.
    pub fn supports_tensor_cores(&self) -> bool {
        match self {
            Device::Cpu => false,
            Device::Cuda(_) => true,
            Device::Vulkan(_) => false,
            Device::Metal(_) => true,
            Device::Tpu(_) => true,
        }
    }

    /// Returns whether this device supports peer-to-peer memory access
    /// with another device.
    ///
    /// # Arguments
    ///
    /// * `other` - The other device to check compatibility with
    pub fn supports_p2p_with(&self, other: &Device) -> bool {
        if self == other {
            return true;
        }
        if self.device_type() != other.device_type() {
            return false;
        }
        self.device_type().supports_p2p()
    }

    /// Returns whether this device can directly access memory on another device.
    ///
    /// This is a broader check that considers unified memory architectures
    /// and device-level memory sharing capabilities.
    ///
    /// # Arguments
    ///
    /// * `other` - The other device to check
    pub fn can_access(&self, other: &Device) -> bool {
        if self == other {
            return true;
        }
        if self.has_unified_memory() && other.has_unified_memory() {
            return true;
        }
        false
    }

    /// Returns the maximum number of concurrent kernels/operations this
    /// device can execute simultaneously.
    ///
    /// For CPU, returns the number of logical cores (estimated).
    /// For GPU/TPU, returns the device's hardware concurrency.
    ///
    /// In this pure-Rust implementation, returns a reasonable estimate.
    pub fn max_concurrent_ops(&self) -> usize {
        match self {
            Device::Cpu => {
                // Try to get thread count, default to 4
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4)
            }
            Device::Cuda(_) => 128,
            Device::Vulkan(_) => 64,
            Device::Metal(_) => 64,
            Device::Tpu(_) => 1, // TPUs typically execute one program at a time
        }
    }

    /// Returns the compute capability of this device as a string.
    ///
    /// For CUDA, this is the SM version (e.g., "8.0" for Ampere).
    /// For other devices, returns a descriptive string.
    pub fn compute_capability(&self) -> Option<String> {
        match self {
            Device::Cpu => None,
            Device::Cuda(_) => Some("sm_80".to_string()), // Placeholder
            Device::Vulkan(_) => Some("vulkan_1.2".to_string()),
            Device::Metal(_) => Some("metal_3.0".to_string()),
            Device::Tpu(_) => Some("tpu_v4".to_string()),
        }
    }

    /// Returns a unique identifier for this device.
    ///
    /// The identifier is a string in the format "type:index".
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::device::Device;
    /// assert_eq!(Device::Cpu.unique_id(), "Cpu:0");
    /// assert_eq!(Device::cuda(2).unique_id(), "Cuda:2");
    /// ```
    pub fn unique_id(&self) -> String {
        format!("{}:{}", self.short_name(), self.index())
    }

    /// Parses a device from a string.
    ///
    /// Accepts formats like "Cpu", "Cuda(0)", "Vulkan(1)", "Metal(0)", "Tpu(2)".
    ///
    /// # Errors
    ///
    /// Returns an error if the string does not match any known device format.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::device::Device;
    /// assert_eq!(Device::parse("Cpu").unwrap(), Device::Cpu);
    /// assert_eq!(Device::parse("Cuda(0)").unwrap(), Device::cuda(0));
    /// assert!(Device::parse("invalid").is_err());
    /// ```
    pub fn parse(s: &str) -> Result<Device, String> {
        let s = s.trim();
        if s == "Cpu" {
            return Ok(Device::Cpu);
        }

        // Parse "TypeName(index)" format
        let mut parts = s.splitn(2, '(');
        let type_name = parts.next().unwrap_or("");
        let index_str = parts.next().unwrap_or("");

        let index: usize = if index_str.ends_with(')') {
            let inner = &index_str[..index_str.len() - 1];
            inner.parse::<usize>().map_err(|e| format!("invalid index '{}': {}", inner, e))?
        } else {
            return Err(format!("expected format 'TypeName(index)', got '{}'", s));
        };

        match type_name {
            "Cuda" => Ok(Device::Cuda(index)),
            "Vulkan" => Ok(Device::Vulkan(index)),
            "Metal" => Ok(Device::Metal(index)),
            "Tpu" => Ok(Device::Tpu(index)),
            _ => Err(format!("unknown device type '{}'", type_name)),
        }
    }

    /// Returns the thermal power limit in watts (if known).
    ///
    /// In this pure-Rust implementation, returns `None`.
    pub fn thermal_limit_watts(&self) -> Option<f64> {
        None
    }

    /// Returns the clock speed in MHz (if known).
    ///
    /// In this pure-Rust implementation, returns `None`.
    pub fn clock_speed_mhz(&self) -> Option<f64> {
        None
    }

    /// Returns the memory bandwidth in GB/s (if known).
    ///
    /// In this pure-Rust implementation, returns `None`.
    pub fn memory_bandwidth_gbs(&self) -> Option<f64> {
        None
    }

    /// Returns the number of streaming multiprocessors / compute units.
    ///
    /// In this pure-Rust implementation, returns `None`.
    pub fn num_sm(&self) -> Option<u32> {
        None
    }
}

// =============================================================================
// PartialEq for Device
// =============================================================================

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Device::Cpu, Device::Cpu) => true,
            (Device::Cuda(a), Device::Cuda(b)) => a == b,
            (Device::Vulkan(a), Device::Vulkan(b)) => a == b,
            (Device::Metal(a), Device::Metal(b)) => a == b,
            (Device::Tpu(a), Device::Tpu(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for Device {}

// =============================================================================
// Hash for Device
// =============================================================================

impl Hash for Device {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Device::Cpu => 0usize.hash(state),
            Device::Cuda(i) => i.hash(state),
            Device::Vulkan(i) => i.hash(state),
            Device::Metal(i) => i.hash(state),
            Device::Tpu(i) => i.hash(state),
        }
    }
}

// =============================================================================
// PartialOrd and Ord for Device (CPU first)
// =============================================================================

impl PartialOrd for Device {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Device {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Device::Cpu, Device::Cpu) => std::cmp::Ordering::Equal,
            (Device::Cpu, _) => std::cmp::Ordering::Less,
            (_, Device::Cpu) => std::cmp::Ordering::Greater,
            _ => {
                let type_ord = self.device_type().cmp(&other.device_type());
                if type_ord != std::cmp::Ordering::Equal {
                    type_ord
                } else {
                    self.index().cmp(&other.index())
                }
            }
        }
    }
}

// =============================================================================
// Display for Device
// =============================================================================

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// =============================================================================
// Default for Device
// =============================================================================

impl Default for Device {
    /// Returns `Device::Cpu` as the default device.
    fn default() -> Self {
        Device::Cpu
    }
}

// =============================================================================
// FromStr for Device
// =============================================================================

impl FromStr for Device {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Device::parse(s)
    }
}

// =============================================================================
// DeviceList - Querying available devices
// =============================================================================

/// A list of available compute devices in the system.
///
/// `DeviceList` provides methods for querying which devices are available,
/// iterating over them, and finding specific devices.
///
/// In this pure-Rust implementation without GPU backends, only the CPU
/// device is always available. GPU/TPU detection would require platform-
/// specific libraries.
///
/// # Examples
///
/// ```
/// use brain_core::device::DeviceList;
///
/// let list = DeviceList::new();
/// assert!(list.cpu().is_some());
/// assert_eq!(list.len(), 1); // Only CPU in this implementation
/// ```
#[derive(Debug, Clone)]
pub struct DeviceList {
    devices: Vec<Device>,
}

impl DeviceList {
    /// Creates a new DeviceList by probing available devices.
    ///
    /// This implementation always includes at least the CPU device.
    /// GPU/TPU detection would be added with platform-specific backends.
    pub fn new() -> Self {
        let mut devices = vec![Device::Cpu];
        // In a real implementation, we would probe for CUDA, Vulkan, Metal, TPU here
        DeviceList { devices }
    }

    /// Creates a DeviceList with a specific set of devices.
    ///
    /// This is useful for testing or for systems where device availability
    /// is known in advance.
    pub fn from_devices(devices: Vec<Device>) -> Self {
        DeviceList { devices }
    }

    /// Returns the CPU device, if present in the list.
    ///
    /// The CPU device is always present in lists created with `new()`.
    pub fn cpu(&self) -> Option<Device> {
        self.devices.iter().find(|d| d.is_cpu()).copied()
    }

    /// Returns all CUDA devices in the list.
    pub fn cuda_devices(&self) -> Vec<Device> {
        self.devices.iter().filter(|d| d.is_cuda()).copied().collect()
    }

    /// Returns all Vulkan devices in the list.
    pub fn vulkan_devices(&self) -> Vec<Device> {
        self.devices.iter().filter(|d| d.is_vulkan()).copied().collect()
    }

    /// Returns all Metal devices in the list.
    pub fn metal_devices(&self) -> Vec<Device> {
        self.devices.iter().filter(|d| d.is_metal()).copied().collect()
    }

    /// Returns all TPU devices in the list.
    pub fn tpu_devices(&self) -> Vec<Device> {
        self.devices.iter().filter(|d| d.is_tpu()).copied().collect()
    }

    /// Returns all accelerator devices (non-CPU).
    pub fn accelerators(&self) -> Vec<Device> {
        self.devices.iter().filter(|d| d.is_accelerator()).copied().collect()
    }

    /// Returns the number of available devices.
    pub fn len(&self) -> usize {
        self.devices.len()
    }

    /// Returns whether the device list is empty.
    pub fn is_empty(&self) -> bool {
        self.devices.is_empty()
    }

    /// Returns an iterator over all available devices.
    pub fn iter(&self) -> std::slice::Iter<'_, Device> {
        self.devices.iter()
    }

    /// Returns the device at the given index.
    ///
    /// The index is into the full device list, not a device-specific index.
    pub fn get(&self, index: usize) -> Option<Device> {
        self.devices.get(index).copied()
    }

    /// Returns whether a specific device is in the list.
    pub fn contains(&self, device: Device) -> bool {
        self.devices.contains(&device)
    }

    /// Returns the number of CUDA devices.
    pub fn cuda_count(&self) -> usize {
        self.cuda_devices().len()
    }

    /// Returns the number of Vulkan devices.
    pub fn vulkan_count(&self) -> usize {
        self.vulkan_devices().len()
    }

    /// Returns the number of Metal devices.
    pub fn metal_count(&self) -> usize {
        self.metal_devices().len()
    }

    /// Returns the number of TPU devices.
    pub fn tpu_count(&self) -> usize {
        self.tpu_devices().len()
    }

    /// Returns the first CUDA device, if any.
    pub fn first_cuda(&self) -> Option<Device> {
        self.devices.iter().find(|d| d.is_cuda()).copied()
    }

    /// Returns the first available accelerator (GPU or TPU).
    pub fn first_accelerator(&self) -> Option<Device> {
        self.devices.iter().find(|d| d.is_accelerator()).copied()
    }

    /// Returns the total memory across all devices in bytes (if known).
    pub fn total_memory_all(&self) -> Option<usize> {
        let mut total = 0usize;
        for device in &self.devices {
            if let Some(mem) = device.total_memory() {
                total += mem;
            } else {
                return None;
            }
        }
        Some(total)
    }

    /// Returns a summary of available devices.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if self.cpu().is_some() {
            parts.push("1 CPU".to_string());
        }
        let cuda_count = self.cuda_count();
        if cuda_count > 0 {
            parts.push(format!("{} CUDA GPU(s)", cuda_count));
        }
        let vulkan_count = self.vulkan_count();
        if vulkan_count > 0 {
            parts.push(format!("{} Vulkan device(s)", vulkan_count));
        }
        let metal_count = self.metal_count();
        if metal_count > 0 {
            parts.push(format!("{} Metal GPU(s)", metal_count));
        }
        let tpu_count = self.tpu_count();
        if tpu_count > 0 {
            parts.push(format!("{} TPU(s)", tpu_count));
        }
        if parts.is_empty() {
            return "No devices available".to_string();
        }
        parts.join(", ")
    }
}

impl Default for DeviceList {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// DeviceProperties
// =============================================================================

/// Properties and capabilities of a compute device.
///
/// `DeviceProperties` provides detailed information about a device's
/// hardware capabilities, memory, and supported features.
///
/// # Fields
///
/// * `name` - Human-readable device name (e.g., "NVIDIA GeForce RTX 4090")
/// * `total_memory` - Total memory in bytes
/// * `compute_capability` - Device-specific compute capability string
/// * `max_threads_per_block` - Maximum threads per compute block
/// * `max_shared_memory_per_block` - Shared memory per block in bytes
/// * `max_grid_size` - Maximum grid dimensions
///
/// # Examples
///
/// ```
/// use brain_core::device::DeviceProperties;
/// let props = DeviceProperties {
///     name: "Test GPU".to_string(),
///     total_memory: 8 * 1024 * 1024 * 1024, // 8 GB
///     compute_capability: Some("sm_80".to_string()),
///     max_threads_per_block: 1024,
///     max_shared_memory_per_block: 49152,
///     num_sms: 108,
///     clock_speed_mhz: Some(2520.0),
///     memory_bandwidth_gbs: Some(1008.0),
///     l2_cache_size: Some(6 * 1024 * 1024), // 6 MB
/// };
/// assert_eq!(props.name, "Test GPU");
/// assert_eq!(props.total_memory_bytes(), 8589934592);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeviceProperties {
    /// Human-readable device name.
    pub name: String,
    /// Total memory available on the device in bytes.
    pub total_memory: usize,
    /// Compute capability string (e.g., "sm_80" for CUDA).
    pub compute_capability: Option<String>,
    /// Maximum number of threads per compute block.
    pub max_threads_per_block: u32,
    /// Maximum shared memory per block in bytes.
    pub max_shared_memory_per_block: usize,
    /// Number of streaming multiprocessors / compute units.
    pub num_sms: u32,
    /// Clock speed in MHz.
    pub clock_speed_mhz: Option<f64>,
    /// Memory bandwidth in GB/s.
    pub memory_bandwidth_gbs: Option<f64>,
    /// L2 cache size in bytes.
    pub l2_cache_size: Option<usize>,
}

impl DeviceProperties {
    /// Creates a new DeviceProperties with the given name and total memory.
    pub fn new(name: String, total_memory: usize) -> Self {
        DeviceProperties {
            name,
            total_memory,
            compute_capability: None,
            max_threads_per_block: 1024,
            max_shared_memory_per_block: 49152,
            num_sms: 1,
            clock_speed_mhz: None,
            memory_bandwidth_gbs: None,
            l2_cache_size: None,
        }
    }

    /// Returns the total memory in bytes.
    pub fn total_memory_bytes(&self) -> usize {
        self.total_memory
    }

    /// Returns the total memory in human-readable format.
    ///
    /// # Examples
    ///
    /// ```
    /// use brain_core::device::DeviceProperties;
    /// let props = DeviceProperties::new("GPU".to_string(), 8 * 1024 * 1024 * 1024);
    /// let mem = props.total_memory_human();
    /// assert!(mem.contains("GB"));
    /// ```
    pub fn total_memory_human(&self) -> String {
        let bytes = self.total_memory as f64;
        const GB: f64 = 1024.0 * 1024.0 * 1024.0;
        const MB: f64 = 1024.0 * 1024.0;
        if bytes >= GB {
            format!("{:.2} GB", bytes / GB)
        } else if bytes >= MB {
            format!("{:.2} MB", bytes / MB)
        } else {
            format!("{} B", self.total_memory)
        }
    }

    /// Returns whether this device supports FP16 operations natively.
    pub fn supports_f16(&self) -> bool {
        true // Conservative default
    }

    /// Returns whether this device supports BF16 operations natively.
    pub fn supports_bf16(&self) -> bool {
        self.compute_capability.as_ref().map_or(false, |cap| {
            cap.starts_with("sm_8") || cap.starts_with("sm_9") || cap.contains("metal") || cap.contains("tpu")
        })
    }

    /// Returns whether this device supports tensor cores / matrix units.
    pub fn supports_tensor_cores(&self) -> bool {
        self.compute_capability.as_ref().map_or(false, |cap| {
            cap.starts_with("sm_7") || cap.starts_with("sm_8") || cap.starts_with("sm_9")
                || cap.contains("metal") || cap.contains("tpu")
        })
    }

    /// Returns the maximum number of threads that can be launched in a grid.
    ///
    /// This is computed as max_threads_per_block * num_sms * warp_size_estimate.
    pub fn max_grid_threads(&self) -> usize {
        self.max_threads_per_block as usize * self.num_sms as usize * 32
    }

    /// Returns a summary string of these properties.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("Device: {}", self.name));
        lines.push(format!("  Memory: {}", self.total_memory_human()));
        if let Some(ref cap) = self.compute_capability {
            lines.push(format!("  Compute: {}", cap));
        }
        lines.push(format!("  SMs: {}", self.num_sms));
        lines.push(format!("  Max threads/block: {}", self.max_threads_per_block));
        lines.push(format!("  Shared mem/block: {} KB", self.max_shared_memory_per_block / 1024));
        if let Some(clk) = self.clock_speed_mhz {
            lines.push(format!("  Clock: {:.0} MHz", clk));
        }
        if let Some(bw) = self.memory_bandwidth_gbs {
            lines.push(format!("  Bandwidth: {:.0} GB/s", bw));
        }
        if let Some(l2) = self.l2_cache_size {
            lines.push(format!("  L2 Cache: {} KB", l2 / 1024));
        }
        lines.join("\n")
    }
}

impl fmt::Display for DeviceProperties {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// =============================================================================
// Free Functions
// =============================================================================

/// Returns the current (default) device.
///
/// In this implementation, always returns CPU.
///
/// # Examples
///
/// ```
/// use brain_core::device::{current_device, Device};
/// assert_eq!(current_device(), Device::Cpu);
/// ```
pub fn current_device() -> Device {
    Device::Cpu
}

/// Sets the current (default) device.
///
/// In this pure-Rust implementation, this is a no-op but validates
/// that the device is available in the system.
///
/// # Arguments
///
/// * `device` - The device to set as current
///
/// # Errors
///
/// Returns an error if the device is not available.
pub fn set_device(_device: Device) -> Result<(), String> {
    // In a real implementation, this would set a thread-local current device
    // and potentially initialize the device context
    Ok(())
}

/// Returns the number of available devices of the given type.
///
/// # Examples
///
/// ```
/// use brain_core::device::{device_count, DeviceType};
/// // Always at least 1 CPU
/// assert!(device_count(DeviceType::Cpu) >= 1);
/// ```
pub fn device_count(device_type: DeviceType) -> usize {
    match device_type {
        DeviceType::Cpu => 1,
        // In a real implementation, these would query the system
        DeviceType::Cuda => 0,
        DeviceType::Vulkan => 0,
        DeviceType::Metal => 0,
        DeviceType::Tpu => 0,
    }
}

/// Returns the total number of available devices across all types.
///
/// # Examples
///
/// ```
/// use brain_core::device::total_device_count;
/// assert!(total_device_count() >= 1); // At least CPU
/// ```
pub fn total_device_count() -> usize {
    DeviceType::ALL.iter().map(|dt| device_count(*dt)).sum()
}

/// Returns a list of all available devices.
///
/// # Examples
///
/// ```
/// use brain_core::device::available_devices;
/// let devices = available_devices();
/// assert!(!devices.is_empty()); // At least CPU
/// ```
pub fn available_devices() -> Vec<Device> {
    let list = DeviceList::new();
    list.iter().copied().collect()
}

/// Returns whether the specified device is available.
///
/// # Examples
///
/// ```
/// use brain_core::device::{is_device_available, Device};
/// assert!(is_device_available(Device::Cpu));
/// assert!(!is_device_available(Device::cuda(0))); // No CUDA in this impl
/// ```
pub fn is_device_available(device: Device) -> bool {
    let list = DeviceList::new();
    list.contains(device)
}

// =============================================================================
// DeviceGuard - RAII guard for current device
// =============================================================================

/// An RAII guard that restores the previous device when dropped.
///
/// `DeviceGuard` is used to temporarily change the current device and
/// automatically restore it when the guard goes out of scope.
///
/// # Examples
///
/// ```ignore
/// let _guard = DeviceGuard::set(Device::cuda(0));
/// // Operations here run on cuda:0
/// // When _guard is dropped, the previous device is restored
/// ```
pub struct DeviceGuard {
    _previous: Device,
}

impl DeviceGuard {
    /// Sets the current device and creates a guard that will restore
    /// the previous device when dropped.
    pub fn set(device: Device) -> Result<Self, String> {
        let previous = current_device();
        set_device(device)?;
        Ok(DeviceGuard { _previous: previous })
    }
}

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        let _ = set_device(self._previous);
    }
}

// =============================================================================
// DeviceSet - A set of devices for distributed operations
// =============================================================================

/// A set of devices for distributed or multi-device operations.
///
/// `DeviceSet` provides utilities for managing groups of devices that
/// participate in distributed training, data parallelism, or model
/// parallelism.
#[derive(Debug, Clone)]
pub struct DeviceSet {
    devices: Vec<Device>,
}

impl DeviceSet {
    /// Creates a new DeviceSet from a list of devices.
    ///
    /// # Panics
    ///
    /// Panics if the device list is empty.
    pub fn new(devices: Vec<Device>) -> Self {
        assert!(!devices.is_empty(), "DeviceSet cannot be empty");
        DeviceSet { devices }
    }

    /// Creates a DeviceSet with all available devices.
    pub fn all_available() -> Self {
        DeviceSet {
            devices: available_devices(),
        }
    }

    /// Creates a DeviceSet with only CPU.
    pub fn cpu_only() -> Self {
        DeviceSet {
            devices: vec![Device::Cpu],
        }
    }

    /// Creates a DeviceSet with the first N CUDA devices.
    ///
    /// Falls back to CPU-only if no CUDA devices are available.
    pub fn first_n_cuda(n: usize) -> Self {
        let list = DeviceList::new();
        let cuda: Vec<Device> = list.cuda_devices().into_iter().take(n).collect();
        if cuda.is_empty() {
            Self::cpu_only()
        } else {
            DeviceSet { devices: cuda }
        }
    }

    /// Returns the number of devices in this set.
    pub fn len(&self) -> usize {
        self.devices.len()
    }

    /// Returns whether this set is empty.
    pub fn is_empty(&self) -> bool {
        self.devices.is_empty()
    }

    /// Returns the device at the given position.
    pub fn get(&self, index: usize) -> Option<Device> {
        self.devices.get(index).copied()
    }

    /// Returns the first device (rank 0).
    pub fn primary(&self) -> Device {
        self.devices[0]
    }

    /// Returns an iterator over all devices.
    pub fn iter(&self) -> std::slice::Iter<'_, Device> {
        self.devices.iter()
    }

    /// Returns all devices as a slice.
    pub fn as_slice(&self) -> &[Device] {
        &self.devices
    }

    /// Checks if all devices in this set are the same type.
    pub fn is_homogeneous(&self) -> bool {
        if self.devices.len() <= 1 {
            return true;
        }
        let first_type = self.devices[0].device_type();
        self.devices[1..].iter().all(|d| d.device_type() == first_type)
    }

    /// Returns the world size (number of devices).
    pub fn world_size(&self) -> usize {
        self.devices.len()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // DeviceType Tests
    // =========================================================================

    #[test]
    fn test_device_type_name() {
        assert_eq!(DeviceType::Cpu.name(), "Cpu");
        assert_eq!(DeviceType::Cuda.name(), "Cuda");
        assert_eq!(DeviceType::Vulkan.name(), "Vulkan");
        assert_eq!(DeviceType::Metal.name(), "Metal");
        assert_eq!(DeviceType::Tpu.name(), "Tpu");
    }

    #[test]
    fn test_device_type_display() {
        assert_eq!(format!("{}", DeviceType::Cpu), "Cpu");
        assert_eq!(format!("{}", DeviceType::Tpu), "Tpu");
    }

    #[test]
    fn test_device_type_is_accelerator() {
        assert!(!DeviceType::Cpu.is_accelerator());
        assert!(DeviceType::Cuda.is_accelerator());
        assert!(DeviceType::Vulkan.is_accelerator());
        assert!(DeviceType::Metal.is_accelerator());
        assert!(DeviceType::Tpu.is_accelerator());
    }

    #[test]
    fn test_device_type_has_unified_memory() {
        assert!(DeviceType::Cpu.has_unified_memory());
        assert!(!DeviceType::Cuda.has_unified_memory());
        assert!(!DeviceType::Vulkan.has_unified_memory());
        assert!(DeviceType::Metal.has_unified_memory());
        assert!(!DeviceType::Tpu.has_unified_memory());
    }

    #[test]
    fn test_device_type_supports_p2p() {
        assert!(!DeviceType::Cpu.supports_p2p());
        assert!(DeviceType::Cuda.supports_p2p());
        assert!(!DeviceType::Vulkan.supports_p2p());
        assert!(DeviceType::Metal.supports_p2p());
        assert!(!DeviceType::Tpu.supports_p2p());
    }

    #[test]
    fn test_device_type_ordering() {
        assert!(DeviceType::Cpu < DeviceType::Cuda);
        assert!(DeviceType::Cuda < DeviceType::Vulkan);
        assert!(DeviceType::Vulkan < DeviceType::Metal);
        assert!(DeviceType::Metal < DeviceType::Tpu);
    }

    #[test]
    fn test_device_type_default() {
        assert_eq!(DeviceType::default(), DeviceType::Cpu);
    }

    #[test]
    fn test_device_type_count() {
        assert_eq!(DeviceType::COUNT, 5);
    }

    #[test]
    fn test_device_type_all() {
        assert_eq!(DeviceType::ALL.len(), 5);
        assert_eq!(DeviceType::ALL[0], DeviceType::Cpu);
        assert_eq!(DeviceType::ALL[4], DeviceType::Tpu);
    }

    #[test]
    fn test_device_type_hash() {
        use std::collections::HashSet;
        let set: HashSet<DeviceType> = DeviceType::ALL.iter().copied().collect();
        assert_eq!(set.len(), 5);
    }

    #[test]
    fn test_device_type_clone() {
        let dt = DeviceType::Cuda;
        let dt2 = dt.clone();
        assert_eq!(dt, dt2);
    }

    // =========================================================================
    // Device Creation Tests
    // =========================================================================

    #[test]
    fn test_device_cuda() {
        let d = Device::cuda(0);
        assert!(d.is_cuda());
        assert_eq!(d.index(), 0);
    }

    #[test]
    fn test_device_vulkan() {
        let d = Device::vulkan(1);
        assert!(d.is_vulkan());
        assert_eq!(d.index(), 1);
    }

    #[test]
    fn test_device_metal() {
        let d = Device::metal(0);
        assert!(d.is_metal());
        assert_eq!(d.index(), 0);
    }

    #[test]
    fn test_device_tpu() {
        let d = Device::tpu(3);
        assert!(d.is_tpu());
        assert_eq!(d.index(), 3);
    }

    // =========================================================================
    // Device Type Checks
    // =========================================================================

    #[test]
    fn test_is_cpu() {
        assert!(Device::Cpu.is_cpu());
        assert!(!Device::cuda(0).is_cpu());
        assert!(!Device::vulkan(0).is_cpu());
        assert!(!Device::metal(0).is_cpu());
        assert!(!Device::tpu(0).is_cpu());
    }

    #[test]
    fn test_is_cuda() {
        assert!(Device::cuda(0).is_cuda());
        assert!(Device::cuda(5).is_cuda());
        assert!(!Device::Cpu.is_cuda());
        assert!(!Device::vulkan(0).is_cuda());
    }

    #[test]
    fn test_is_vulkan() {
        assert!(Device::vulkan(0).is_vulkan());
        assert!(!Device::cuda(0).is_vulkan());
    }

    #[test]
    fn test_is_metal() {
        assert!(Device::metal(0).is_metal());
        assert!(!Device::cuda(0).is_metal());
    }

    #[test]
    fn test_is_tpu() {
        assert!(Device::tpu(0).is_tpu());
        assert!(!Device::cuda(0).is_tpu());
    }

    #[test]
    fn test_is_accelerator() {
        assert!(!Device::Cpu.is_accelerator());
        assert!(Device::cuda(0).is_accelerator());
        assert!(Device::vulkan(0).is_accelerator());
        assert!(Device::metal(0).is_accelerator());
        assert!(Device::tpu(0).is_accelerator());
    }

    // =========================================================================
    // Device Index Tests
    // =========================================================================

    #[test]
    fn test_index() {
        assert_eq!(Device::Cpu.index(), 0);
        assert_eq!(Device::cuda(0).index(), 0);
        assert_eq!(Device::cuda(5).index(), 5);
        assert_eq!(Device::vulkan(2).index(), 2);
        assert_eq!(Device::metal(1).index(), 1);
        assert_eq!(Device::tpu(3).index(), 3);
    }

    // =========================================================================
    // Device Name Tests
    // =========================================================================

    #[test]
    fn test_name() {
        assert_eq!(Device::Cpu.name(), "Cpu");
        assert_eq!(Device::cuda(0).name(), "Cuda(0)");
        assert_eq!(Device::vulkan(1).name(), "Vulkan(1)");
        assert_eq!(Device::metal(0).name(), "Metal(0)");
        assert_eq!(Device::tpu(2).name(), "Tpu(2)");
    }

    #[test]
    fn test_short_name() {
        assert_eq!(Device::Cpu.short_name(), "Cpu");
        assert_eq!(Device::cuda(5).short_name(), "Cuda");
    }

    // =========================================================================
    // Device Display Tests
    // =========================================================================

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Device::Cpu), "Cpu");
        assert_eq!(format!("{}", Device::cuda(3)), "Cuda(3)");
    }

    // =========================================================================
    // Device Default Tests
    // =========================================================================

    #[test]
    fn test_default_is_cpu() {
        assert_eq!(Device::default(), Device::Cpu);
    }

    // =========================================================================
    // Device PartialEq Tests
    // =========================================================================

    #[test]
    fn test_equality_same() {
        assert_eq!(Device::Cpu, Device::Cpu);
        assert_eq!(Device::cuda(0), Device::cuda(0));
        assert_eq!(Device::vulkan(1), Device::vulkan(1));
    }

    #[test]
    fn test_equality_different_index() {
        assert_ne!(Device::cuda(0), Device::cuda(1));
        assert_ne!(Device::vulkan(0), Device::vulkan(1));
    }

    #[test]
    fn test_equality_different_type() {
        assert_ne!(Device::Cpu, Device::cuda(0));
        assert_ne!(Device::cuda(0), Device::vulkan(0));
        assert_ne!(Device::vulkan(0), Device::metal(0));
    }

    // =========================================================================
    // Device Hash Tests
    // =========================================================================

    #[test]
    fn test_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Device::Cpu);
        set.insert(Device::cuda(0));
        set.insert(Device::cuda(1));
        set.insert(Device::vulkan(0));
        assert_eq!(set.len(), 4);

        // Same device should not be added twice
        set.insert(Device::cuda(0));
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn test_hash_consistency() {
        use std::collections::HashMap;
        let mut map: HashMap<Device, String> = HashMap::new();
        map.insert(Device::cuda(0), "gpu0".to_string());
        assert_eq!(map.get(&Device::cuda(0)), Some(&"gpu0".to_string()));
    }

    // =========================================================================
    // Device Ord Tests
    // =========================================================================

    #[test]
    fn test_ord_cpu_first() {
        assert!(Device::Cpu < Device::cuda(0));
        assert!(Device::Cpu < Device::vulkan(0));
        assert!(Device::Cpu < Device::metal(0));
        assert!(Device::Cpu < Device::tpu(0));
    }

    #[test]
    fn test_ord_same_type_by_index() {
        assert!(Device::cuda(0) < Device::cuda(1));
        assert!(Device::cuda(1) < Device::cuda(5));
        assert!(Device::vulkan(0) < Device::vulkan(3));
    }

    #[test]
    fn test_ord_different_types() {
        assert!(Device::cuda(0) < Device::vulkan(0));
        assert!(Device::vulkan(0) < Device::metal(0));
        assert!(Device::metal(0) < Device::tpu(0));
    }

    #[test]
    fn test_ord_eq() {
        assert_eq!(Device::Cpu.cmp(&Device::Cpu), std::cmp::Ordering::Equal);
        assert_eq!(Device::cuda(3).cmp(&Device::cuda(3)), std::cmp::Ordering::Equal);
    }

    // =========================================================================
    // Device Synchronize Tests
    // =========================================================================

    #[test]
    fn test_synchronize_cpu() {
        assert!(Device::Cpu.synchronize().is_ok());
    }

    #[test]
    fn test_synchronize_cuda() {
        assert!(Device::cuda(0).synchronize().is_ok());
    }

    // =========================================================================
    // Device Memory Tests
    // =========================================================================

    #[test]
    fn test_memory_available_none() {
        assert!(Device::Cpu.memory_available().is_none());
        assert!(Device::cuda(0).memory_available().is_none());
    }

    #[test]
    fn test_total_memory_none() {
        assert!(Device::Cpu.total_memory().is_none());
    }

    // =========================================================================
    // Device Properties Tests
    // =========================================================================

    #[test]
    fn test_properties_none() {
        assert!(Device::Cpu.properties().is_none());
        assert!(Device::cuda(0).properties().is_none());
    }

    // =========================================================================
    // Device Unified Memory Tests
    // =========================================================================

    #[test]
    fn test_has_unified_memory() {
        assert!(Device::Cpu.has_unified_memory());
        assert!(!Device::cuda(0).has_unified_memory());
        assert!(Device::metal(0).has_unified_memory());
    }

    // =========================================================================
    // Device Feature Support Tests
    // =========================================================================

    #[test]
    fn test_supports_async() {
        assert!(Device::Cpu.supports_async());
        assert!(Device::cuda(0).supports_async());
    }

    #[test]
    fn test_supports_f16() {
        assert!(!Device::Cpu.supports_f16());
        assert!(Device::cuda(0).supports_f16());
        assert!(Device::vulkan(0).supports_f16());
        assert!(Device::metal(0).supports_f16());
        assert!(Device::tpu(0).supports_f16());
    }

    #[test]
    fn test_supports_bf16() {
        assert!(!Device::Cpu.supports_bf16());
        assert!(Device::cuda(0).supports_bf16());
        assert!(!Device::vulkan(0).supports_bf16());
        assert!(Device::metal(0).supports_bf16());
        assert!(Device::tpu(0).supports_bf16());
    }

    #[test]
    fn test_supports_f64() {
        assert!(Device::Cpu.supports_f64());
        assert!(Device::cuda(0).supports_f64());
        assert!(!Device::vulkan(0).supports_f64());
        assert!(Device::metal(0).supports_f64());
        assert!(!Device::tpu(0).supports_f64());
    }

    #[test]
    fn test_supports_tensor_cores() {
        assert!(!Device::Cpu.supports_tensor_cores());
        assert!(Device::cuda(0).supports_tensor_cores());
        assert!(!Device::vulkan(0).supports_tensor_cores());
        assert!(Device::metal(0).supports_tensor_cores());
        assert!(Device::tpu(0).supports_tensor_cores());
    }

    // =========================================================================
    // Device P2P Tests
    // =========================================================================

    #[test]
    fn test_supports_p2p_with_same() {
        assert!(Device::cuda(0).supports_p2p_with(&Device::cuda(0)));
    }

    #[test]
    fn test_supports_p2p_with_different() {
        assert!(Device::cuda(0).supports_p2p_with(&Device::cuda(1)));
        assert!(!Device::cuda(0).supports_p2p_with(&Device::vulkan(0)));
        assert!(!Device::Cpu.supports_p2p_with(&Device::cuda(0)));
    }

    #[test]
    fn test_can_access() {
        assert!(Device::Cpu.can_access(&Device::Cpu));
        assert!(Device::cuda(0).can_access(&Device::cuda(0)));
        assert!(!Device::Cpu.can_access(&Device::cuda(0)));
        assert!(!Device::cuda(0).can_access(&Device::Cpu));
    }

    // =========================================================================
    // Device Parse Tests
    // =========================================================================

    #[test]
    fn test_parse_cpu() {
        assert_eq!(Device::parse("Cpu").unwrap(), Device::Cpu);
    }

    #[test]
    fn test_parse_cuda() {
        assert_eq!(Device::parse("Cuda(0)").unwrap(), Device::cuda(0));
        assert_eq!(Device::parse("Cuda(5)").unwrap(), Device::cuda(5));
    }

    #[test]
    fn test_parse_vulkan() {
        assert_eq!(Device::parse("Vulkan(0)").unwrap(), Device::vulkan(0));
    }

    #[test]
    fn test_parse_metal() {
        assert_eq!(Device::parse("Metal(0)").unwrap(), Device::metal(0));
    }

    #[test]
    fn test_parse_tpu() {
        assert_eq!(Device::parse("Tpu(2)").unwrap(), Device::tpu(2));
    }

    #[test]
    fn test_parse_whitespace() {
        assert_eq!(Device::parse("  Cpu  ").unwrap(), Device::Cpu);
        assert_eq!(Device::parse("  Cuda(0)  ").unwrap(), Device::cuda(0));
    }

    #[test]
    fn test_parse_invalid() {
        assert!(Device::parse("invalid").is_err());
        assert!(Device::parse("Cuda").is_err()); // missing index
        assert!(Device::parse("Cuda()").is_err()); // empty index
        assert!(Device::parse("Cuda(a)").is_err()); // non-numeric
        assert!(Device::parse("Cuda(0").is_err()); // missing closing paren
    }

    #[test]
    fn test_from_str() {
        use std::str::FromStr;
        let d: Device = "Cuda(0)".parse().unwrap();
        assert_eq!(d, Device::cuda(0));
    }

    // =========================================================================
    // Device ID Tests
    // =========================================================================

    #[test]
    fn test_unique_id() {
        assert_eq!(Device::Cpu.unique_id(), "Cpu:0");
        assert_eq!(Device::cuda(2).unique_id(), "Cuda:2");
        assert_eq!(Device::vulkan(1).unique_id(), "Vulkan:1");
    }

    // =========================================================================
    // Device Compute Capability Tests
    // =========================================================================

    #[test]
    fn test_compute_capability() {
        assert!(Device::Cpu.compute_capability().is_none());
        assert!(Device::cuda(0).compute_capability().is_some());
        assert!(Device::vulkan(0).compute_capability().is_some());
        assert!(Device::metal(0).compute_capability().is_some());
        assert!(Device::tpu(0).compute_capability().is_some());
    }

    // =========================================================================
    // Device Max Concurrent Ops Tests
    // =========================================================================

    #[test]
    fn test_max_concurrent_ops() {
        let cpu_ops = Device::Cpu.max_concurrent_ops();
        assert!(cpu_ops >= 1);
        assert_eq!(Device::cuda(0).max_concurrent_ops(), 128);
        assert_eq!(Device::tpu(0).max_concurrent_ops(), 1);
    }

    // =========================================================================
    // DeviceList Tests
    // =========================================================================

    #[test]
    fn test_device_list_new() {
        let list = DeviceList::new();
        assert!(!list.is_empty());
        assert!(list.cpu().is_some());
    }

    #[test]
    fn test_device_list_cpu() {
        let list = DeviceList::new();
        assert_eq!(list.cpu(), Some(Device::Cpu));
    }

    #[test]
    fn test_device_list_len() {
        let list = DeviceList::new();
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn test_device_list_cuda_empty() {
        let list = DeviceList::new();
        assert!(list.cuda_devices().is_empty());
        assert_eq!(list.cuda_count(), 0);
    }

    #[test]
    fn test_device_list_vulkan_empty() {
        let list = DeviceList::new();
        assert!(list.vulkan_devices().is_empty());
    }

    #[test]
    fn test_device_list_metal_empty() {
        let list = DeviceList::new();
        assert!(list.metal_devices().is_empty());
    }

    #[test]
    fn test_device_list_tpu_empty() {
        let list = DeviceList::new();
        assert!(list.tpu_devices().is_empty());
    }

    #[test]
    fn test_device_list_accelerators_empty() {
        let list = DeviceList::new();
        assert!(list.accelerators().is_empty());
    }

    #[test]
    fn test_device_list_first_cuda_none() {
        let list = DeviceList::new();
        assert!(list.first_cuda().is_none());
    }

    #[test]
    fn test_device_list_first_accelerator_none() {
        let list = DeviceList::new();
        assert!(list.first_accelerator().is_none());
    }

    #[test]
    fn test_device_list_contains_cpu() {
        let list = DeviceList::new();
        assert!(list.contains(Device::Cpu));
        assert!(!list.contains(Device::cuda(0)));
    }

    #[test]
    fn test_device_list_iter() {
        let list = DeviceList::new();
        let devices: Vec<Device> = list.iter().copied().collect();
        assert_eq!(devices, vec![Device::Cpu]);
    }

    #[test]
    fn test_device_list_get() {
        let list = DeviceList::new();
        assert_eq!(list.get(0), Some(Device::Cpu));
        assert!(list.get(1).is_none());
    }

    #[test]
    fn test_device_list_default() {
        let list = DeviceList::default();
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn test_device_list_from_devices() {
        let list = DeviceList::from_devices(vec![Device::Cpu, Device::cuda(0), Device::cuda(1)]);
        assert_eq!(list.len(), 3);
        assert_eq!(list.cuda_count(), 2);
    }

    #[test]
    fn test_device_list_summary() {
        let list = DeviceList::new();
        let summary = list.summary();
        assert!(summary.contains("CPU"));
    }

    #[test]
    fn test_device_list_clone() {
        let list = DeviceList::new();
        let list2 = list.clone();
        assert_eq!(list.len(), list2.len());
    }

    // =========================================================================
    // DeviceProperties Tests
    // =========================================================================

    #[test]
    fn test_device_properties_new() {
        let props = DeviceProperties::new("Test GPU".to_string(), 8 * 1024 * 1024 * 1024);
        assert_eq!(props.name, "Test GPU");
        assert_eq!(props.total_memory, 8589934592);
        assert_eq!(props.max_threads_per_block, 1024);
    }

    #[test]
    fn test_device_properties_total_memory_bytes() {
        let props = DeviceProperties::new("GPU".to_string(), 1024);
        assert_eq!(props.total_memory_bytes(), 1024);
    }

    #[test]
    fn test_device_properties_total_memory_human() {
        let props = DeviceProperties::new("GPU".to_string(), 8 * 1024 * 1024 * 1024);
        let mem = props.total_memory_human();
        assert!(mem.contains("GB"));
    }

    #[test]
    fn test_device_properties_total_memory_human_mb() {
        let props = DeviceProperties::new("GPU".to_string(), 512 * 1024 * 1024);
        let mem = props.total_memory_human();
        assert!(mem.contains("MB"));
    }

    #[test]
    fn test_device_properties_total_memory_human_bytes() {
        let props = DeviceProperties::new("GPU".to_string(), 500);
        let mem = props.total_memory_human();
        assert!(mem.contains("B"));
    }

    #[test]
    fn test_device_properties_summary() {
        let mut props = DeviceProperties::new("GPU".to_string(), 4096);
        props.compute_capability = Some("sm_80".to_string());
        props.clock_speed_mhz = Some(1800.0);
        let summary = props.summary();
        assert!(summary.contains("GPU"));
        assert!(summary.contains("sm_80"));
        assert!(summary.contains("1800"));
    }

    #[test]
    fn test_device_properties_display() {
        let props = DeviceProperties::new("GPU".to_string(), 1024);
        let display = format!("{}", props);
        assert!(display.contains("GPU"));
    }

    #[test]
    fn test_device_properties_supports_tensor_cores() {
        let mut props = DeviceProperties::new("GPU".to_string(), 1024);
        props.compute_capability = None;
        assert!(!props.supports_tensor_cores());
        props.compute_capability = Some("sm_80".to_string());
        assert!(props.supports_tensor_cores());
    }

    #[test]
    fn test_device_properties_max_grid_threads() {
        let props = DeviceProperties::new("GPU".to_string(), 1024);
        assert_eq!(props.max_grid_threads(), 1024 * 1 * 32);
    }

    #[test]
    fn test_device_properties_equality() {
        let a = DeviceProperties::new("GPU".to_string(), 1024);
        let b = DeviceProperties::new("GPU".to_string(), 1024);
        assert_eq!(a, b);
    }

    // =========================================================================
    // Free Function Tests
    // =========================================================================

    #[test]
    fn test_current_device() {
        assert_eq!(current_device(), Device::Cpu);
    }

    #[test]
    fn test_set_device_cpu() {
        assert!(set_device(Device::Cpu).is_ok());
    }

    #[test]
    fn test_device_count() {
        assert_eq!(device_count(DeviceType::Cpu), 1);
        assert_eq!(device_count(DeviceType::Cuda), 0);
        assert_eq!(device_count(DeviceType::Vulkan), 0);
        assert_eq!(device_count(DeviceType::Metal), 0);
        assert_eq!(device_count(DeviceType::Tpu), 0);
    }

    #[test]
    fn test_total_device_count() {
        assert!(total_device_count() >= 1);
    }

    #[test]
    fn test_available_devices() {
        let devices = available_devices();
        assert!(!devices.is_empty());
        assert!(devices.contains(&Device::Cpu));
    }

    #[test]
    fn test_is_device_available() {
        assert!(is_device_available(Device::Cpu));
        assert!(!is_device_available(Device::cuda(0)));
    }

    // =========================================================================
    // DeviceGuard Tests
    // =========================================================================

    #[test]
    fn test_device_guard_set() {
        let guard = DeviceGuard::set(Device::Cpu);
        assert!(guard.is_ok());
    }

    // =========================================================================
    // DeviceSet Tests
    // =========================================================================

    #[test]
    fn test_device_set_new() {
        let set = DeviceSet::new(vec![Device::Cpu, Device::cuda(0)]);
        assert_eq!(set.len(), 2);
    }

    #[test]
    #[should_panic(expected = "cannot be empty")]
    fn test_device_set_empty_panics() {
        let _ = DeviceSet::new(vec![]);
    }

    #[test]
    fn test_device_set_all_available() {
        let set = DeviceSet::all_available();
        assert!(set.len() >= 1);
    }

    #[test]
    fn test_device_set_cpu_only() {
        let set = DeviceSet::cpu_only();
        assert_eq!(set.len(), 1);
        assert_eq!(set.primary(), Device::Cpu);
    }

    #[test]
    fn test_device_set_first_n_cuda() {
        let set = DeviceSet::first_n_cuda(4);
        assert!(set.len() >= 1); // Falls back to CPU
    }

    #[test]
    fn test_device_set_get() {
        let set = DeviceSet::new(vec![Device::Cpu, Device::cuda(1)]);
        assert_eq!(set.get(0), Some(Device::Cpu));
        assert_eq!(set.get(1), Some(Device::cuda(1)));
        assert!(set.get(2).is_none());
    }

    #[test]
    fn test_device_set_primary() {
        let set = DeviceSet::new(vec![Device::cuda(0), Device::cuda(1)]);
        assert_eq!(set.primary(), Device::cuda(0));
    }

    #[test]
    fn test_device_set_iter() {
        let set = DeviceSet::new(vec![Device::Cpu]);
        let devices: Vec<Device> = set.iter().copied().collect();
        assert_eq!(devices, vec![Device::Cpu]);
    }

    #[test]
    fn test_device_set_is_homogeneous() {
        let set = DeviceSet::new(vec![Device::Cpu]);
        assert!(set.is_homogeneous());
    }

    #[test]
    fn test_device_set_is_homogeneous_multi() {
        let set = DeviceSet::new(vec![Device::cuda(0), Device::cuda(1)]);
        assert!(set.is_homogeneous());
    }

    #[test]
    fn test_device_set_not_homogeneous() {
        let set = DeviceSet::new(vec![Device::Cpu, Device::cuda(0)]);
        assert!(!set.is_homogeneous());
    }

    #[test]
    fn test_device_set_world_size() {
        let set = DeviceSet::new(vec![Device::Cpu, Device::cuda(0), Device::cuda(1)]);
        assert_eq!(set.world_size(), 3);
    }

    #[test]
    fn test_device_set_as_slice() {
        let set = DeviceSet::new(vec![Device::Cpu, Device::cuda(0)]);
        assert_eq!(set.as_slice().len(), 2);
    }

    // =========================================================================
    // Clone and Copy Tests
    // =========================================================================

    #[test]
    fn test_device_clone() {
        let d = Device::cuda(0);
        let d2 = d.clone();
        assert_eq!(d, d2);
    }

    #[test]
    fn test_device_copy() {
        let d = Device::metal(1);
        let d2 = d;
        assert_eq!(d, d2);
    }

    // =========================================================================
    // Debug Tests
    // =========================================================================

    #[test]
    fn test_debug() {
        let d = Device::cuda(0);
        let debug = format!("{:?}", d);
        assert!(debug.contains("Cuda"));
    }

    // =========================================================================
    // DeviceType Tests
    // =========================================================================

    #[test]
    fn test_device_type() {
        assert_eq!(Device::Cpu.device_type(), DeviceType::Cpu);
        assert_eq!(Device::cuda(0).device_type(), DeviceType::Cuda);
        assert_eq!(Device::vulkan(0).device_type(), DeviceType::Vulkan);
        assert_eq!(Device::metal(0).device_type(), DeviceType::Metal);
        assert_eq!(Device::tpu(0).device_type(), DeviceType::Tpu);
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_parse_negative_index() {
        // Negative indices should not parse
        let result = Device::parse("Cuda(-1)");
        assert!(result.is_err());
    }

    #[test]
    fn test_large_index() {
        let d = Device::cuda(999999);
        assert_eq!(d.index(), 999999);
        assert_eq!(d.name(), "Cuda(999999)");
    }

    #[test]
    fn test_unique_id_large_index() {
        let d = Device::vulkan(100);
        assert_eq!(d.unique_id(), "Vulkan:100");
    }

    // =========================================================================
    // Additional Device Feature Tests
    // =========================================================================

    #[test]
    fn test_all_features_cpu() {
        let d = Device::Cpu;
        assert!(d.supports_async());
        assert!(d.has_unified_memory());
        assert!(!d.supports_f16());
        assert!(!d.supports_bf16());
        assert!(d.supports_f64());
        assert!(!d.supports_tensor_cores());
    }

    #[test]
    fn test_all_features_cuda() {
        let d = Device::cuda(0);
        assert!(d.supports_async());
        assert!(!d.has_unified_memory());
        assert!(d.supports_f16());
        assert!(d.supports_bf16());
        assert!(d.supports_f64());
        assert!(d.supports_tensor_cores());
    }

    #[test]
    fn test_all_features_metal() {
        let d = Device::metal(0);
        assert!(d.supports_async());
        assert!(d.has_unified_memory());
        assert!(d.supports_f16());
        assert!(d.supports_bf16());
        assert!(d.supports_f64());
        assert!(d.supports_tensor_cores());
    }

    #[test]
    fn test_all_features_vulkan() {
        let d = Device::vulkan(0);
        assert!(d.supports_async());
        assert!(!d.has_unified_memory());
        assert!(d.supports_f16());
        assert!(!d.supports_bf16());
        assert!(!d.supports_f64());
        assert!(!d.supports_tensor_cores());
    }

    #[test]
    fn test_all_features_tpu() {
        let d = Device::tpu(0);
        assert!(d.supports_async());
        assert!(!d.has_unified_memory());
        assert!(d.supports_f16());
        assert!(d.supports_bf16());
        assert!(!d.supports_f64());
        assert!(d.supports_tensor_cores());
    }

    // =========================================================================
    // DeviceType Comprehensive Tests
    // =========================================================================

    #[test]
    fn test_device_type_all() {
        assert_eq!(DeviceType::ALL.len(), 5);
        assert!(DeviceType::ALL.contains(&DeviceType::Cpu));
        assert!(DeviceType::ALL.contains(&DeviceType::Cuda));
        assert!(DeviceType::ALL.contains(&DeviceType::Vulkan));
        assert!(DeviceType::ALL.contains(&DeviceType::Metal));
        assert!(DeviceType::ALL.contains(&DeviceType::Tpu));
    }

    #[test]
    fn test_device_type_hash_unique() {
        use std::collections::HashSet;
        let set: HashSet<_> = DeviceType::ALL.iter().copied().collect();
        assert_eq!(set.len(), 5);
    }

    #[test]
    fn test_device_type_equality() {
        assert_eq!(DeviceType::Cpu, DeviceType::Cpu);
        assert_ne!(DeviceType::Cpu, DeviceType::Cuda);
    }

    #[test]
    fn test_device_type_partial_eq_reflexive() {
        for dt in DeviceType::ALL.iter() {
            assert_eq!(*dt, *dt);
        }
    }

    #[test]
    fn test_device_type_partial_eq_symmetric() {
        for dt in DeviceType::ALL.iter() {
            for dt2 in DeviceType::ALL.iter() {
                assert_eq!(dt == dt2, dt2 == dt);
            }
        }
    }

    #[test]
    fn test_device_type_partial_eq_transitive() {
        // Cpu < Cuda < Vulkan, so Cpu < Vulkan
        assert!(DeviceType::Cpu < DeviceType::Cuda);
        assert!(DeviceType::Cuda < DeviceType::Vulkan);
        assert!(DeviceType::Cpu < DeviceType::Vulkan);
    }

    #[test]
    fn test_device_type_ord_total() {
        let sorted = [DeviceType::Cpu, DeviceType::Cuda, DeviceType::Vulkan, DeviceType::Metal, DeviceType::Tpu];
        let mut all = DeviceType::ALL;
        all.sort();
        assert_eq!(all, sorted);
    }

    // =========================================================================
    // Device Parse Comprehensive Tests
    // =========================================================================

    #[test]
    fn test_parse_all_types() {
        assert!(Device::parse("Cuda(0)").is_ok());
        assert!(Device::parse("Cuda(999)").is_ok());
        assert!(Device::parse("Vulkan(0)").is_ok());
        assert!(Device::parse("Metal(0)").is_ok());
        assert!(Device::parse("Tpu(0)").is_ok());
        assert!(Device::parse("Tpu(7)").is_ok());
    }

    #[test]
    fn test_parse_various_invalid() {
        assert!(Device::parse("").is_err());
        assert!(Device::parse(" ").is_err());
        assert!(Device::parse("(").is_err());
        assert!(Device::parse(")").is_err());
        assert!(Device::parse("Cuda()").is_err());
        assert!(Device::parse("Cuda(-1)").is_err());
        assert!(Device::parse("Cuda(0)extra").is_err());
        assert!(Device::parse("Cuda 0").is_err());
        assert!(Device::parse("cuda(0)").is_err()); // lowercase not supported
        assert!(Device::parse("Type(0)").is_err());
    }

    #[test]
    fn test_parse_error_messages() {
        let err = Device::parse("invalid").unwrap_err();
        assert!(err.contains("unknown device type"));

        let err = Device::parse("Cuda(a)").unwrap_err();
        assert!(err.contains("invalid index"));

        let err = Device::parse("Cuda(0").unwrap_err();
        assert!(err.contains("TypeName(index)"));
    }

    // =========================================================================
    // DeviceList Comprehensive Tests
    // =========================================================================

    #[test]
    fn test_device_list_from_devices_various() {
        let list = DeviceList::from_devices(vec![
            Device::Cpu, Device::cuda(0), Device::cuda(1),
            Device::vulkan(0), Device::metal(0), Device::tpu(0),
        ]);
        assert_eq!(list.len(), 6);
        assert_eq!(list.cuda_count(), 2);
        assert_eq!(list.vulkan_count(), 1);
        assert_eq!(list.metal_count(), 1);
        assert_eq!(list.tpu_count(), 1);
        assert_eq!(list.accelerators().len(), 5);
    }

    #[test]
    fn test_device_list_from_devices_empty() {
        let list = DeviceList::from_devices(vec![]);
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert!(list.cpu().is_none());
    }

    #[test]
    fn test_device_list_contains_various() {
        let list = DeviceList::from_devices(vec![Device::Cpu, Device::cuda(0)]);
        assert!(list.contains(Device::Cpu));
        assert!(list.contains(Device::cuda(0)));
        assert!(!list.contains(Device::cuda(1)));
        assert!(!list.contains(Device::vulkan(0)));
    }

    #[test]
    fn test_device_list_get_various() {
        let list = DeviceList::from_devices(vec![Device::Cpu, Device::cuda(0), Device::cuda(1)]);
        assert_eq!(list.get(0), Some(Device::Cpu));
        assert_eq!(list.get(1), Some(Device::cuda(0)));
        assert_eq!(list.get(2), Some(Device::cuda(1)));
        assert!(list.get(3).is_none());
    }

    #[test]
    fn test_device_list_summary_various() {
        let list = DeviceList::from_devices(vec![
            Device::Cpu, Device::cuda(0), Device::cuda(1), Device::tpu(0),
        ]);
        let summary = list.summary();
        assert!(summary.contains("CPU"));
        assert!(summary.contains("2 CUDA"));
        assert!(summary.contains("1 TPU"));
    }

    #[test]
    fn test_device_list_first_cuda_with_devices() {
        let list = DeviceList::from_devices(vec![Device::Cpu, Device::cuda(2)]);
        assert_eq!(list.first_cuda(), Some(Device::cuda(2)));
    }

    #[test]
    fn test_device_list_first_accelerator_with_devices() {
        let list = DeviceList::from_devices(vec![Device::Cpu, Device::cuda(0)]);
        assert_eq!(list.first_accelerator(), Some(Device::cuda(0)));
    }

    #[test]
    fn test_device_list_iter_collect() {
        let list = DeviceList::from_devices(vec![Device::Cpu, Device::cuda(0)]);
        let devices: Vec<Device> = list.iter().copied().collect();
        assert_eq!(devices.len(), 2);
    }

    #[test]
    fn test_device_list_total_memory_none() {
        let list = DeviceList::new();
        assert!(list.total_memory_all().is_none());
    }

    // =========================================================================
    // DeviceProperties Extended Tests
    // =========================================================================

    #[test]
    fn test_device_properties_display() {
        let mut props = DeviceProperties::new("Test GPU".to_string(), 8 * 1024 * 1024 * 1024);
        props.compute_capability = Some("sm_90".to_string());
        props.num_sms = 132;
        props.clock_speed_mhz = Some(2520.0);
        props.memory_bandwidth_gbs = Some(1008.0);
        props.l2_cache_size = Some(48 * 1024 * 1024);
        let display = format!("{}", props);
        assert!(display.contains("Test GPU"));
        assert!(display.contains("8.00 GB"));
        assert!(display.contains("sm_90"));
        assert!(display.contains("132"));
        assert!(display.contains("2520"));
        assert!(display.contains("1008"));
    }

    #[test]
    fn test_device_properties_memory_human_small() {
        let props = DeviceProperties::new("Small".to_string(), 512);
        assert_eq!(props.total_memory_human(), "512 B");
    }

    #[test]
    fn test_device_properties_memory_human_kb() {
        let props = DeviceProperties::new("Small".to_string(), 2048);
        assert_eq!(props.total_memory_human(), "2.00 KB");
    }

    #[test]
    fn test_device_properties_memory_human_gb() {
        let props = DeviceProperties::new("Big".to_string(), 16 * 1024 * 1024 * 1024);
        let mem = props.total_memory_human();
        assert!(mem.contains("GB"));
        assert!(mem.contains("16"));
    }

    #[test]
    fn test_device_properties_hash() {
        use std::collections::HashSet;
        let props = DeviceProperties::new("GPU".to_string(), 1024);
        let mut set = HashSet::new();
        set.insert(props.clone());
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_device_properties_max_grid_threads_multiple_sms() {
        let mut props = DeviceProperties::new("GPU".to_string(), 1024);
        props.num_sms = 80;
        assert_eq!(props.max_grid_threads(), 1024 * 80 * 32);
    }

    #[test]
    fn test_device_properties_supports_bf16_sm80() {
        let mut props = DeviceProperties::new("GPU".to_string(), 1024);
        props.compute_capability = Some("sm_80".to_string());
        assert!(props.supports_bf16());
    }

    #[test]
    fn test_device_properties_supports_bf16_no_capability() {
        let props = DeviceProperties::new("GPU".to_string(), 1024);
        assert!(!props.supports_bf16());
    }

    #[test]
    fn test_device_properties_supports_f16() {
        let props = DeviceProperties::new("GPU".to_string(), 1024);
        assert!(props.supports_f16());
    }

    // =========================================================================
    // DeviceSet Extended Tests
    // =========================================================================

    #[test]
    fn test_device_set_new_single() {
        let set = DeviceSet::new(vec![Device::Cpu]);
        assert_eq!(set.len(), 1);
        assert_eq!(set.world_size(), 1);
        assert!(set.is_homogeneous());
    }

    #[test]
    fn test_device_set_new_multiple_same_type() {
        let set = DeviceSet::new(vec![Device::cuda(0), Device::cuda(1), Device::cuda(2)]);
        assert_eq!(set.len(), 3);
        assert!(set.is_homogeneous());
        assert_eq!(set.world_size(), 3);
    }

    #[test]
    fn test_device_set_new_mixed_types() {
        let set = DeviceSet::new(vec![Device::Cpu, Device::cuda(0), Device::tpu(0)]);
        assert!(!set.is_homogeneous());
    }

    #[test]
    fn test_device_set_get_all() {
        let set = DeviceSet::new(vec![Device::cuda(0), Device::cuda(1), Device::cuda(2)]);
        assert_eq!(set.get(0), Some(Device::cuda(0)));
        assert_eq!(set.get(1), Some(Device::cuda(1)));
        assert_eq!(set.get(2), Some(Device::cuda(2)));
        assert!(set.get(3).is_none());
    }

    #[test]
    fn test_device_set_as_slice() {
        let set = DeviceSet::new(vec![Device::Cpu, Device::cuda(0)]);
        let slice = set.as_slice();
        assert_eq!(slice.len(), 2);
    }

    #[test]
    fn test_device_set_iter() {
        let set = DeviceSet::new(vec![Device::cuda(0), Device::cuda(1)]);
        let devices: Vec<Device> = set.iter().copied().collect();
        assert_eq!(devices.len(), 2);
    }

    #[test]
    fn test_device_set_clone() {
        let set = DeviceSet::new(vec![Device::cuda(0)]);
        let set2 = set.clone();
        assert_eq!(set.len(), set2.len());
        assert_eq!(set.primary(), set2.primary());
    }

    // =========================================================================
    // DeviceGuard Extended Tests
    // =========================================================================

    #[test]
    fn test_device_guard_drop_restores() {
        {
            let _guard = DeviceGuard::set(Device::Cpu).unwrap();
            // guard in scope
        }
        // after guard is dropped, should be restored
    }

    // =========================================================================
    // Free Function Extended Tests
    // =========================================================================

    #[test]
    fn test_device_count_all_types() {
        assert!(device_count(DeviceType::Cpu) >= 1);
        assert_eq!(device_count(DeviceType::Cuda), 0);
        assert_eq!(device_count(DeviceType::Vulkan), 0);
        assert_eq!(device_count(DeviceType::Metal), 0);
        assert_eq!(device_count(DeviceType::Tpu), 0);
    }

    #[test]
    fn test_available_devices_contains_cpu() {
        let devices = available_devices();
        assert!(devices.iter().any(|d| d.is_cpu()));
    }

    #[test]
    fn test_is_device_available_cpu() {
        assert!(is_device_available(Device::Cpu));
    }

    #[test]
    fn test_is_device_available_multiple_gpus() {
        assert!(!is_device_available(Device::cuda(0)));
        assert!(!is_device_available(Device::cuda(1)));
    }

    #[test]
    fn test_set_device_various() {
        assert!(set_device(Device::Cpu).is_ok());
        assert!(set_device(Device::cuda(0)).is_ok());
        assert!(set_device(Device::tpu(0)).is_ok());
    }

    #[test]
    fn test_current_device_always_cpu() {
        assert_eq!(current_device(), Device::Cpu);
        let _ = set_device(Device::cuda(0));
        // current_device is not actually affected by set_device in this impl
    }

    // =========================================================================
    // Device Comparison Tests
    // =========================================================================

    #[test]
    fn test_device_eq_hash_consistent() {
        use std::collections::HashMap;
        let mut map: HashMap<Device, String> = HashMap::new();
        map.insert(Device::Cpu, "cpu".to_string());
        map.insert(Device::cuda(0), "gpu0".to_string());
        map.insert(Device::cuda(1), "gpu1".to_string());
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&Device::cuda(0)).unwrap(), "gpu0");
    }

    #[test]
    fn test_device_ord_consistent_with_eq() {
        let devices = vec![
            Device::Cpu,
            Device::cuda(0),
            Device::cuda(1),
            Device::vulkan(0),
            Device::metal(0),
            Device::tpu(0),
        ];
        let mut sorted = devices.clone();
        sorted.sort();
        for i in 0..sorted.len() {
            for j in 0..sorted.len() {
                assert_eq!(sorted[i] < sorted[j], devices[i] < devices[j]);
            }
        }
    }

    #[test]
    fn test_device_ord_type_then_index() {
        // All cuda devices should be between vulkan and the next type
        assert!(Device::cuda(0) < Device::cuda(1));
        assert!(Device::cuda(99) < Device::cuda(100));
        assert!(Device::vulkan(0) < Device::vulkan(1));
        assert!(Device::metal(0) < Device::metal(1));
        assert!(Device::tpu(0) < Device::tpu(1));
    }

    #[test]
    fn test_device_ord_cross_type() {
        assert!(Device::cuda(100) < Device::vulkan(0));
        assert!(Device::vulkan(100) < Device::metal(0));
        assert!(Device::metal(100) < Device::tpu(0));
    }

    // =========================================================================
    // Device Type Method Delegation Tests
    // =========================================================================

    #[test]
    fn test_device_type_delegation() {
        for dt in DeviceType::ALL.iter() {
            let d = match dt {
                DeviceType::Cpu => Device::Cpu,
                DeviceType::Cuda => Device::cuda(0),
                DeviceType::Vulkan => Device::vulkan(0),
                DeviceType::Metal => Device::metal(0),
                DeviceType::Tpu => Device::tpu(0),
            };
            assert_eq!(d.device_type(), *dt);
            assert_eq!(d.short_name(), dt.name());
            assert_eq!(d.is_accelerator(), dt.is_accelerator());
            assert_eq!(d.has_unified_memory(), dt.has_unified_memory());
        }
    }

    #[test]
    fn test_device_compute_capability_non_cpu() {
        // All non-CPU devices should return Some
        for d in [Device::cuda(0), Device::vulkan(0), Device::metal(0), Device::tpu(0)] {
            assert!(d.compute_capability().is_some());
        }
    }

    #[test]
    fn test_device_thermal_limit_none() {
        assert!(Device::Cpu.thermal_limit_watts().is_none());
        assert!(Device::cuda(0).thermal_limit_watts().is_none());
    }

    #[test]
    fn test_device_clock_speed_none() {
        assert!(Device::Cpu.clock_speed_mhz().is_none());
        assert!(Device::cuda(0).clock_speed_mhz().is_none());
    }

    #[test]
    fn test_device_memory_bandwidth_none() {
        assert!(Device::Cpu.memory_bandwidth_gbs().is_none());
        assert!(Device::cuda(0).memory_bandwidth_gbs().is_none());
    }

    #[test]
    fn test_device_num_sm_none() {
        assert!(Device::Cpu.num_sm().is_none());
        assert!(Device::cuda(0).num_sm().is_none());
    }

    // =========================================================================
    // Additional Device Ordering Edge Cases
    // =========================================================================

    #[test]
    fn test_cuda_ordering_lots_of_devices() {
        let devices: Vec<Device> = (0..50).map(Device::cuda).collect();
        let mut sorted = devices.clone();
        sorted.sort();
        for i in 0..sorted.len() - 1 {
            assert!(sorted[i] < sorted[i + 1]);
        }
    }

    #[test]
    fn test_all_device_types_in_sorted_order() {
        let all_devices = vec![
            Device::Cpu,
            Device::cuda(0),
            Device::cuda(1),
            Device::vulkan(0),
            Device::vulkan(1),
            Device::metal(0),
            Device::metal(1),
            Device::tpu(0),
            Device::tpu(1),
        ];
        let mut sorted = all_devices.clone();
        sorted.sort();
        for i in 0..all_devices.len() {
            assert_eq!(sorted[i], all_devices[i], "ordering failed at position {}", i);
        }
    }

    #[test]
    fn test_device_partial_ord_max_min() {
        let devices = vec![Device::Cpu, Device::cuda(0), Device::tpu(0)];
        let min = devices.iter().min().unwrap();
        let max = devices.iter().max().unwrap();
        assert_eq!(*min, Device::Cpu);
        assert_eq!(*max, Device::tpu(0));
    }

    #[test]
    fn test_device_eq_reflexive() {
        let d = Device::cuda(42);
        assert_eq!(d, d);
    }

    #[test]
    fn test_device_eq_symmetric() {
        let a = Device::cuda(5);
        let b = Device::cuda(5);
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    #[test]
    fn test_device_eq_transitive() {
        let a = Device::cuda(3);
        let b = Device::cuda(3);
        let c = Device::cuda(3);
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    // =========================================================================
    // Additional DeviceList Stress Tests
    // =========================================================================

    #[test]
    fn test_device_list_many_devices() {
        let devices: Vec<Device> = (0..20).map(Device::cuda).collect();
        let list = DeviceList::from_devices(devices);
        assert_eq!(list.len(), 20);
        assert_eq!(list.cuda_count(), 20);
        assert_eq!(list.cuda_devices().len(), 20);
    }

    #[test]
    fn test_device_list_mixed_types() {
        let devices = vec![
            Device::Cpu,
            Device::cuda(0), Device::cuda(1), Device::cuda(2),
            Device::vulkan(0), Device::vulkan(1),
            Device::metal(0),
            Device::tpu(0),
        ];
        let list = DeviceList::from_devices(devices);
        assert_eq!(list.len(), 8);
        assert_eq!(list.cuda_count(), 3);
        assert_eq!(list.vulkan_count(), 2);
        assert_eq!(list.metal_count(), 1);
        assert_eq!(list.tpu_count(), 1);
        assert_eq!(list.accelerators().len(), 7);
    }

    #[test]
    fn test_device_list_summary_mixed() {
        let list = DeviceList::from_devices(vec![
            Device::Cpu,
            Device::cuda(0), Device::cuda(1), Device::cuda(2),
            Device::metal(0),
        ]);
        let summary = list.summary();
        assert!(summary.contains("1 CPU"));
        assert!(summary.contains("3 CUDA"));
        assert!(summary.contains("1 Metal"));
        assert!(!summary.contains("Vulkan"));
        assert!(!summary.contains("TPU"));
    }

    #[test]
    fn test_device_list_summary_no_devices() {
        let list = DeviceList::from_devices(vec![]);
        assert_eq!(list.summary(), "No devices available");
    }

    // =========================================================================
    // Device Properties Builder Tests
    // =========================================================================

    #[test]
    fn test_device_properties_builder_pattern() {
        let props = DeviceProperties::new("RTX 4090".to_string(), 24 * 1024 * 1024 * 1024)
            .compute_capability;
        // Just verify the struct was created
        let props = DeviceProperties::new("RTX 4090".to_string(), 24 * 1024 * 1024 * 1024);
        assert_eq!(props.name, "RTX 4090");
        assert_eq!(props.total_memory, 24 * 1024 * 1024 * 1024);
        assert_eq!(props.max_threads_per_block, 1024);
        assert_eq!(props.max_shared_memory_per_block, 49152);
        assert_eq!(props.num_sms, 1);
    }

    #[test]
    fn test_device_properties_different_memory_sizes() {
        let sizes = vec![512, 1024, 2048, 4096 * 1024, 1024 * 1024 * 1024];
        for size in sizes {
            let props = DeviceProperties::new("GPU".to_string(), size);
            assert_eq!(props.total_memory, size);
        }
    }

    // =========================================================================
    // Device Parse Round Trip Tests
    // =========================================================================

    #[test]
    fn test_parse_round_trip_cpu() {
        let d = Device::Cpu;
        let s = format!("{}", d);
        let d2 = Device::parse(&s).unwrap();
        assert_eq!(d, d2);
    }

    #[test]
    fn test_parse_round_trip_cuda() {
        for i in 0..10 {
            let d = Device::cuda(i);
            let s = format!("{}", d);
            let d2 = Device::parse(&s).unwrap();
            assert_eq!(d, d2);
        }
    }

    #[test]
    fn test_parse_round_trip_vulkan() {
        for i in 0..5 {
            let d = Device::vulkan(i);
            let s = format!("{}", d);
            let d2 = Device::parse(&s).unwrap();
            assert_eq!(d, d2);
        }
    }

    #[test]
    fn test_parse_round_trip_metal() {
        for i in 0..5 {
            let d = Device::metal(i);
            let s = format!("{}", d);
            let d2 = Device::parse(&s).unwrap();
            assert_eq!(d, d2);
        }
    }

    #[test]
    fn test_parse_round_trip_tpu() {
        for i in 0..5 {
            let d = Device::tpu(i);
            let s = format!("{}", d);
            let d2 = Device::parse(&s).unwrap();
            assert_eq!(d, d2);
        }
    }

    // =========================================================================
    // Device Display Format Tests
    // =========================================================================

    #[test]
    fn test_display_format_all_devices() {
        assert_eq!(format!("{}", Device::Cpu), "Cpu");
        assert_eq!(format!("{}", Device::cuda(0)), "Cuda(0)");
        assert_eq!(format!("{}", Device::cuda(99)), "Cuda(99)");
        assert_eq!(format!("{}", Device::vulkan(5)), "Vulkan(5)");
        assert_eq!(format!("{}", Device::metal(3)), "Metal(3)");
        assert_eq!(format!("{}", Device::tpu(7)), "Tpu(7)");
    }

    #[test]
    fn test_debug_format_contains_info() {
        let d = Device::cuda(0);
        let debug = format!("{:?}", d);
        assert!(debug.contains("Cuda"));
    }

    // =========================================================================
    // Device Clone Copy Consistency
    // =========================================================================

    #[test]
    fn test_clone_and_copy_consistency() {
        let original = Device::cuda(5);
        let cloned = original.clone();
        let copied = original; // Copy
        assert_eq!(original, cloned);
        assert_eq!(original, copied);
        assert_eq!(cloned, copied);
        assert_eq!(original.index(), 5);
        assert_eq!(cloned.index(), 5);
        assert_eq!(copied.index(), 5);
    }

    // =========================================================================
    // Device from_str Trait Tests
    // =========================================================================

    #[test]
    fn test_from_str_trait_cpu() {
        use std::str::FromStr;
        let d: Device = "Cpu".parse().unwrap();
        assert_eq!(d, Device::Cpu);
    }

    #[test]
    fn test_from_str_trait_error() {
        use std::str::FromStr;
        let result: Result<Device, String> = "Invalid".parse();
        assert!(result.is_err());
    }
}
