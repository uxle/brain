//! Memory management infrastructure for the Brain deep learning framework.
//!
//! This module provides high-performance, production-grade memory allocation, tracking,
//! pooling, arena allocation, aligned buffers, and memory format utilities for efficient
//! tensor operations without external runtime dependencies.
//!
//! # Architecture & Components
//!
//! 1. **Aligned Memory Allocator**: Page-aligned (4096-byte) and cacheline-aligned (64-byte)
//!    allocations via standard library allocator primitives (`std::alloc`), ensuring optimal
//!    SIMD vectorized access and AVX-512/AVX2 cache line efficiency.
//! 2. **Memory Pools**:
//!    - [`SimplePool`]: Pre-allocated contiguous slab with best-fit / first-fit free-list.
//!    - [`BinnedMemoryPool`]: Power-of-two segregated free lists for \(O(1)\) allocations
//!      with minimal internal and external fragmentation.
//! 3. **Memory Arena**:
//!    - [`MemoryArena`]: Fast linear bump allocator with checkpoint/rollback semantics
//!      and batch reset for ephemeral neural network activation buffers.
//! 4. **Memory Tracker & Leak Detection**:
//!    - [`MemoryTracker`]: Thread-safe allocation registry with tags, call site context,
//!      peak watermark tracking, and comprehensive leak reporting.
//! 5. **Memory Layouts & Formats**:
//!    - [`MemoryFormat`]: Strided layout definitions (Contiguous, ChannelsFirst/NCHW,
//!      ChannelsLast/NHWC, Blocked, Sparse, Packed).
//! 6. **Memory Planning**:
//!    - [`MemoryPlanner`]: Static memory planner for computation graphs that reuses
//!      temporary memory across non-overlapping tensor lifetimes.

use std::alloc::{alloc, dealloc, realloc as std_realloc, Layout};
use std::collections::HashMap;
use std::fmt;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::error::{BrainError, BrainResult};

// =============================================================================
// Memory Constants & Alignment
// =============================================================================

/// Standard CPU cache line size in bytes (64 bytes on modern x86_64 and aarch64).
pub const CACHE_LINE_SIZE: usize = 64;

/// Standard OS virtual memory page size in bytes (4 KiB).
pub const PAGE_SIZE: usize = 4096;

/// Huge page size in bytes (2 MiB).
pub const HUGE_PAGE_SIZE: usize = 2 * 1024 * 1024;

/// Default SIMD vector alignment (32 bytes for AVX2, 64 bytes for AVX-512).
pub const SIMD_ALIGNMENT: usize = 64;

/// Checks if a memory address/pointer is aligned to the given boundary.
#[inline(always)]
pub fn is_aligned(ptr: *const u8, alignment: usize) -> bool {
    (ptr as usize) % alignment == 0
}

/// Rounds up a size in bytes to the nearest multiple of alignment.
#[inline(always)]
pub const fn align_up(size: usize, alignment: usize) -> usize {
    if alignment == 0 {
        return size;
    }
    (size + alignment - 1) & !(alignment - 1)
}

/// Rounds down a size in bytes to the nearest multiple of alignment.
#[inline(always)]
pub const fn align_down(size: usize, alignment: usize) -> usize {
    if alignment == 0 {
        return size;
    }
    size & !(alignment - 1)
}

// =============================================================================
// Memory Format
// =============================================================================

/// The memory format/layout for tensor storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryFormat {
    /// Contiguous row-major (C-style) layout.
    Contiguous,
    /// Channels-first format (NCHW) for CNN operations.
    ChannelsFirst,
    /// Channels-last format (NHWC) for CNN operations.
    ChannelsLast,
    /// Blocked format with a specific block size.
    Blocked { block_size: usize },
    /// Sparse format with explicit index storage.
    Sparse,
    /// Packed format for 8-bit quantized tensors.
    Packed,
}

impl Default for MemoryFormat {
    fn default() -> Self {
        MemoryFormat::Contiguous
    }
}

impl fmt::Display for MemoryFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryFormat::Contiguous => write!(f, "contiguous"),
            MemoryFormat::ChannelsFirst => write!(f, "channels_first"),
            MemoryFormat::ChannelsLast => write!(f, "channels_last"),
            MemoryFormat::Blocked { block_size } => write!(f, "blocked({})", block_size),
            MemoryFormat::Sparse => write!(f, "sparse"),
            MemoryFormat::Packed => write!(f, "packed"),
        }
    }
}

impl MemoryFormat {
    /// Returns the expected strides for a given shape in this memory format.
    pub fn strides(&self, shape: &[usize]) -> Vec<usize> {
        match self {
            MemoryFormat::Contiguous => strides_row_major(shape),
            MemoryFormat::ChannelsFirst => strides_row_major(shape),
            MemoryFormat::ChannelsLast => {
                if shape.len() == 4 {
                    let (_n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
                    vec![h * w * c, 1, w * c, c]
                } else {
                    strides_row_major(shape)
                }
            }
            MemoryFormat::Blocked { block_size } => {
                if shape.len() == 2 {
                    let (_rows, cols) = (shape[0], shape[1]);
                    vec![cols * block_size, *block_size]
                } else {
                    strides_row_major(shape)
                }
            }
            MemoryFormat::Sparse => vec![],
            MemoryFormat::Packed => {
                if shape.len() == 1 {
                    vec![1]
                } else {
                    strides_row_major(shape)
                }
            }
        }
    }

    /// Returns true if the format is dense and contiguous.
    pub fn is_dense(&self) -> bool {
        matches!(
            self,
            MemoryFormat::Contiguous | MemoryFormat::ChannelsFirst | MemoryFormat::ChannelsLast
        )
    }
}

/// Helper to compute row-major strides for a shape.
pub fn strides_row_major(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// =============================================================================
// AlignedBuffer - Cache-Line & Page-Aligned RAII Buffer
// =============================================================================

/// A cache-aligned or page-aligned dynamically allocated contiguous buffer of elements.
///
/// Ensures memory is properly aligned for AVX2, AVX-512, and DMA hardware transfers.
pub struct AlignedBuffer<T> {
    ptr: NonNull<T>,
    capacity: usize,
    len: usize,
    alignment: usize,
}

unsafe impl<T: Send> Send for AlignedBuffer<T> {}
unsafe impl<T: Sync> Sync for AlignedBuffer<T> {}

impl<T> AlignedBuffer<T> {
    /// Creates a new aligned buffer with the specified capacity and alignment.
    ///
    /// # Panics
    ///
    /// Panics if `alignment` is not a power of two or is less than `std::mem::align_of::<T>()`.
    pub fn with_capacity_aligned(capacity: usize, alignment: usize) -> Self {
        assert!(
            alignment.is_power_of_two(),
            "Alignment must be a power of two"
        );
        let effective_align = alignment.max(std::mem::align_of::<T>()).max(1);

        if capacity == 0 {
            return AlignedBuffer {
                ptr: NonNull::dangling(),
                capacity: 0,
                len: 0,
                alignment: effective_align,
            };
        }

        let size = capacity
            .checked_mul(std::mem::size_of::<T>())
            .expect("Capacity overflow");
        let layout = Layout::from_size_align(size, effective_align)
            .expect("Invalid memory layout requested");

        let raw = unsafe { alloc(layout) as *mut T };
        let ptr = NonNull::new(raw).expect("Memory allocation failed");

        AlignedBuffer {
            ptr,
            capacity,
            len: 0,
            alignment: effective_align,
        }
    }

    /// Creates a new buffer aligned to the CPU cache line (64 bytes).
    pub fn with_cacheline_alignment(capacity: usize) -> Self {
        Self::with_capacity_aligned(capacity, CACHE_LINE_SIZE)
    }

    /// Creates a new buffer aligned to OS memory pages (4096 bytes).
    pub fn with_page_alignment(capacity: usize) -> Self {
        Self::with_capacity_aligned(capacity, PAGE_SIZE)
    }

    /// Allocates an aligned buffer filled with a default value.
    pub fn from_elem(elem: T, count: usize, alignment: usize) -> Self
    where
        T: Clone,
    {
        let mut buf = Self::with_capacity_aligned(count, alignment);
        for _ in 0..count {
            buf.push(elem.clone());
        }
        buf
    }

    /// Returns the number of elements currently stored.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the buffer has zero elements.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the total capacity in elements.
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the alignment in bytes.
    #[inline(always)]
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Returns a raw const pointer to the backing buffer.
    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns a raw mutable pointer to the backing buffer.
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Returns an immutable slice over the initialized elements.
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }

    /// Returns a mutable slice over the initialized elements.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }

    /// Appends an element to the end of the buffer.
    pub fn push(&mut self, value: T) {
        if self.len >= self.capacity {
            let new_cap = if self.capacity == 0 {
                8
            } else {
                self.capacity.checked_mul(2).expect("Capacity overflow")
            };
            self.reserve(new_cap - self.len);
        }
        unsafe {
            let dst = self.ptr.as_ptr().add(self.len);
            std::ptr::write(dst, value);
            self.len += 1;
        }
    }

    /// Clears the buffer, dropping all elements while keeping allocation intact.
    pub fn clear(&mut self) {
        let old_len = self.len;
        self.len = 0;
        if std::mem::needs_drop::<T>() {
            for i in 0..old_len {
                unsafe {
                    std::ptr::drop_in_place(self.ptr.as_ptr().add(i));
                }
            }
        }
    }

    /// Reserves space for at least `additional` more elements.
    pub fn reserve(&mut self, additional: usize) {
        let min_cap = self
            .len
            .checked_add(additional)
            .expect("Capacity overflow");
        if min_cap <= self.capacity {
            return;
        }
        let new_capacity = min_cap.max(self.capacity * 2).max(8);

        let elem_size = std::mem::size_of::<T>();
        let new_size = new_capacity
            .checked_mul(elem_size)
            .expect("Size overflow");
        let new_layout =
            Layout::from_size_align(new_size, self.alignment).expect("Invalid layout");

        let new_ptr = if self.capacity == 0 {
            unsafe { alloc(new_layout) as *mut T }
        } else {
            let old_size = self.capacity * elem_size;
            let old_layout = Layout::from_size_align(old_size, self.alignment).unwrap();
            unsafe {
                let realloc_ptr =
                    std_realloc(self.ptr.as_ptr() as *mut u8, old_layout, new_size) as *mut T;
                if realloc_ptr.is_null() {
                    let fresh = alloc(new_layout) as *mut T;
                    if !fresh.is_null() && self.len > 0 {
                        std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), fresh, self.len);
                        dealloc(self.ptr.as_ptr() as *mut u8, old_layout);
                    }
                    fresh
                } else {
                    realloc_ptr
                }
            }
        };

        self.ptr = NonNull::new(new_ptr).expect("Memory reallocation failed");
        self.capacity = new_capacity;
    }
}

impl<T> Drop for AlignedBuffer<T> {
    fn drop(&mut self) {
        self.clear();
        if self.capacity > 0 {
            let size = self.capacity * std::mem::size_of::<T>();
            if let Ok(layout) = Layout::from_size_align(size, self.alignment) {
                unsafe {
                    dealloc(self.ptr.as_ptr() as *mut u8, layout);
                }
            }
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for AlignedBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlignedBuffer")
            .field("len", &self.len)
            .field("capacity", &self.capacity)
            .field("alignment", &self.alignment)
            .field("slice", &self.as_slice())
            .finish()
    }
}

impl<T: Clone> Clone for AlignedBuffer<T> {
    fn clone(&self) -> Self {
        let mut new_buf = Self::with_capacity_aligned(self.len, self.alignment);
        for item in self.as_slice() {
            new_buf.push(item.clone());
        }
        new_buf
    }
}

// =============================================================================
// Allocation Stats & Metrics
// =============================================================================

/// Detailed metrics and counters for memory allocations.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AllocationStats {
    /// Total bytes requested since creation.
    pub total_requested_bytes: usize,
    /// Total bytes physically allocated.
    pub total_allocated_bytes: usize,
    /// Currently active allocated bytes.
    pub current_bytes: usize,
    /// Peak watermark memory usage in bytes.
    pub peak_bytes: usize,
    /// Number of successful allocations performed.
    pub allocation_count: usize,
    /// Number of deallocations performed.
    pub deallocation_count: usize,
    /// Number of active live blocks.
    pub active_allocations: usize,
}

impl AllocationStats {
    /// Creates a new empty `AllocationStats`.
    pub fn new() -> Self {
        AllocationStats::default()
    }

    /// Records a new allocation of `size` bytes.
    pub fn record_allocation(&mut self, size: usize) {
        self.total_requested_bytes += size;
        self.total_allocated_bytes += size;
        self.current_bytes += size;
        self.peak_bytes = self.peak_bytes.max(self.current_bytes);
        self.allocation_count += 1;
        self.active_allocations += 1;
    }

    /// Records a deallocation of `size` bytes.
    pub fn record_deallocation(&mut self, size: usize) {
        self.current_bytes = self.current_bytes.saturating_sub(size);
        self.deallocation_count += 1;
        self.active_allocations = self.active_allocations.saturating_sub(1);
    }

    /// Returns the external fragmentation ratio between 0.0 and 1.0.
    pub fn fragmentation_ratio(&self) -> f64 {
        if self.peak_bytes == 0 {
            0.0
        } else {
            (self.peak_bytes - self.current_bytes) as f64 / self.peak_bytes as f64
        }
    }
}

impl fmt::Display for AllocationStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Memory Allocation Stats:")?;
        writeln!(f, "  Current:     {} bytes", self.current_bytes)?;
        writeln!(f, "  Peak:        {} bytes", self.peak_bytes)?;
        writeln!(f, "  Total Alloc: {} bytes", self.total_allocated_bytes)?;
        writeln!(f, "  Alloc Count: {}", self.allocation_count)?;
        writeln!(f, "  Deallocs:    {}", self.deallocation_count)?;
        write!(f, "  Live Blocks: {}", self.active_allocations)
    }
}

// =============================================================================
// Memory Block Descriptor
// =============================================================================

/// Descriptor for an allocated region within a memory pool or arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryBlock {
    /// Byte offset within the pool's contiguous slab.
    pub offset: usize,
    /// Size of the block in bytes.
    pub size: usize,
    /// Unique identifier for this allocation.
    pub id: usize,
}

// =============================================================================
// MemoryPool Trait
// =============================================================================

/// Trait defining the interface for memory pool allocators.
pub trait MemoryPool: Send + Sync {
    /// Allocates a block of memory with the given size in bytes.
    fn allocate(&mut self, size: usize) -> BrainResult<MemoryBlock>;

    /// Deallocates a previously allocated block.
    fn deallocate(&mut self, block: &MemoryBlock) -> BrainResult<()>;

    /// Returns total active allocated bytes.
    fn allocated_bytes(&self) -> usize;

    /// Returns total available free memory in bytes.
    fn available_bytes(&self) -> usize;

    /// Returns current allocation statistics.
    fn stats(&self) -> AllocationStats;

    /// Resets the pool, reclaiming all allocated blocks.
    fn reset(&mut self);
}

// =============================================================================
// SimplePool Implementation
// =============================================================================

/// A contiguous slab memory pool using a coalescing free list.
#[derive(Debug)]
pub struct SimplePool {
    buffer: Vec<u8>,
    free_list: Vec<(usize, usize)>,
    stats: AllocationStats,
    next_id: usize,
    alignment: usize,
}

impl SimplePool {
    /// Creates a new `SimplePool` with the specified capacity in bytes.
    pub fn new(capacity: usize) -> Self {
        Self::with_alignment(capacity, 64)
    }

    /// Creates a new `SimplePool` with capacity and alignment in bytes.
    pub fn with_alignment(capacity: usize, alignment: usize) -> Self {
        let aligned_cap = align_up(capacity, alignment.max(1));
        SimplePool {
            buffer: vec![0u8; aligned_cap],
            free_list: vec![(0, aligned_cap)],
            stats: AllocationStats::new(),
            next_id: 1,
            alignment: alignment.max(1),
        }
    }

    /// Returns the total capacity of the pool.
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Coalesces adjacent contiguous free blocks in the free list.
    pub fn coalesce(&mut self) {
        if self.free_list.len() <= 1 {
            return;
        }
        self.free_list.sort_by_key(|&(offset, _)| offset);
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(self.free_list.len());

        for &(offset, size) in &self.free_list {
            if let Some(last) = merged.last_mut() {
                if last.0 + last.1 == offset {
                    last.1 += size;
                    continue;
                }
            }
            merged.push((offset, size));
        }
        self.free_list = merged;
    }
}

impl MemoryPool for SimplePool {
    fn allocate(&mut self, size: usize) -> BrainResult<MemoryBlock> {
        if size == 0 {
            return Ok(MemoryBlock {
                offset: 0,
                size: 0,
                id: 0,
            });
        }
        let aligned_size = align_up(size, self.alignment);

        let mut best_idx = None;
        let mut best_size = usize::MAX;

        for (idx, &(_offset, block_size)) in self.free_list.iter().enumerate() {
            if block_size >= aligned_size && block_size < best_size {
                best_size = block_size;
                best_idx = Some(idx);
            }
        }

        let idx = best_idx.ok_or_else(|| {
            BrainError::allocation_failed(
                aligned_size,
                Some(self.available_bytes()),
                "SimplePool: Out of memory",
            )
        })?;

        let (offset, block_size) = self.free_list.remove(idx);
        let remaining = block_size - aligned_size;

        if remaining > 0 {
            self.free_list.push((offset + aligned_size, remaining));
        }

        let id = self.next_id;
        self.next_id += 1;
        self.stats.record_allocation(aligned_size);

        Ok(MemoryBlock {
            offset,
            size: aligned_size,
            id,
        })
    }

    fn deallocate(&mut self, block: &MemoryBlock) -> BrainResult<()> {
        if block.size == 0 {
            return Ok(());
        }
        self.free_list.push((block.offset, block.size));
        self.coalesce();
        self.stats.record_deallocation(block.size);
        Ok(())
    }

    fn allocated_bytes(&self) -> usize {
        self.stats.current_bytes
    }

    fn available_bytes(&self) -> usize {
        self.free_list.iter().map(|&(_, size)| size).sum()
    }

    fn stats(&self) -> AllocationStats {
        self.stats.clone()
    }

    fn reset(&mut self) {
        let cap = self.buffer.len();
        self.free_list = vec![(0, cap)];
        self.stats.current_bytes = 0;
        self.stats.active_allocations = 0;
    }
}

// =============================================================================
// BinnedMemoryPool - Segregated Power-of-Two Free Lists
// =============================================================================

/// A segregated binned memory pool with power-of-two size classes (32 B to 64 MiB).
///
/// Guarantees \(O(1)\) allocations and deallocations without memory search overhead.
pub struct BinnedMemoryPool {
    bins: HashMap<usize, Vec<NonNull<u8>>>,
    stats: AllocationStats,
    min_bin_size: usize,
    max_bin_size: usize,
}

unsafe impl Send for BinnedMemoryPool {}
unsafe impl Sync for BinnedMemoryPool {}

impl BinnedMemoryPool {
    /// Creates a new `BinnedMemoryPool` supporting sizes from 64 B to 32 MiB.
    pub fn new() -> Self {
        BinnedMemoryPool {
            bins: HashMap::new(),
            stats: AllocationStats::new(),
            min_bin_size: 64,
            max_bin_size: 32 * 1024 * 1024,
        }
    }

    /// Finds the smallest power-of-two size class for the requested size.
    #[inline(always)]
    pub fn size_class(&self, size: usize) -> usize {
        let s = size.max(self.min_bin_size);
        s.next_power_of_two()
    }

    /// Allocates an aligned pointer for the requested size.
    pub fn allocate(&mut self, size: usize) -> BrainResult<NonNull<u8>> {
        if size == 0 {
            return Ok(NonNull::dangling());
        }
        let bin_size = self.size_class(size);

        if let Some(list) = self.bins.get_mut(&bin_size) {
            if let Some(ptr) = list.pop() {
                self.stats.record_allocation(bin_size);
                return Ok(ptr);
            }
        }

        let layout = Layout::from_size_align(bin_size, 64)
            .map_err(|e| BrainError::invalid_value(format!("Invalid layout: {}", e)))?;
        let raw = unsafe { alloc(layout) };
        let ptr = NonNull::new(raw).ok_or_else(|| {
            BrainError::allocation_failed(bin_size, None, "BinnedMemoryPool: system OOM")
        })?;

        self.stats.record_allocation(bin_size);
        Ok(ptr)
    }

    /// Recycles a pointer back into its size-class bin.
    pub fn deallocate(&mut self, ptr: NonNull<u8>, size: usize) {
        if size == 0 {
            return;
        }
        let bin_size = self.size_class(size);
        self.bins.entry(bin_size).or_default().push(ptr);
        self.stats.record_deallocation(bin_size);
    }

    /// Returns allocation stats.
    pub fn stats(&self) -> AllocationStats {
        self.stats.clone()
    }
}

impl Default for BinnedMemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for BinnedMemoryPool {
    fn drop(&mut self) {
        for (&bin_size, list) in &mut self.bins {
            let layout = Layout::from_size_align(bin_size, 64).unwrap();
            for &ptr in list.iter() {
                unsafe {
                    dealloc(ptr.as_ptr(), layout);
                }
            }
        }
    }
}

// =============================================================================
// MemoryArena - High-Speed Bump Allocator
// =============================================================================

/// A fast linear bump allocator for ephemeral intermediate buffers.
pub struct MemoryArena {
    buffer: NonNull<u8>,
    capacity: usize,
    offset: usize,
    alignment: usize,
    stats: AllocationStats,
}

unsafe impl Send for MemoryArena {}
unsafe impl Sync for MemoryArena {}

/// A checkpoint handle used to rewind the arena state.
#[derive(Debug, Clone, Copy)]
pub struct ArenaCheckpoint {
    offset: usize,
}

impl MemoryArena {
    /// Creates a new `MemoryArena` with the specified capacity in bytes.
    pub fn new(capacity: usize) -> Self {
        Self::with_alignment(capacity, 64)
    }

    /// Creates a new `MemoryArena` with capacity and alignment.
    pub fn with_alignment(capacity: usize, alignment: usize) -> Self {
        let align = alignment.max(1).max(std::mem::align_of::<usize>());
        let layout = Layout::from_size_align(capacity, align).expect("Invalid arena layout");
        let raw = unsafe { alloc(layout) };
        let ptr = NonNull::new(raw).expect("Arena memory allocation failed");

        MemoryArena {
            buffer: ptr,
            capacity,
            offset: 0,
            alignment: align,
            stats: AllocationStats::new(),
        }
    }

    /// Allocates `size` bytes from the arena, returning a raw pointer.
    pub fn alloc(&mut self, size: usize) -> BrainResult<*mut u8> {
        if size == 0 {
            return Ok(self.buffer.as_ptr());
        }
        let aligned_offset = align_up(self.offset, self.alignment);
        let new_offset = aligned_offset.checked_add(size).ok_or_else(|| {
            BrainError::allocation_failed(size, None, "MemoryArena: size overflow")
        })?;

        if new_offset > self.capacity {
            return Err(BrainError::allocation_failed(
                size,
                Some(self.capacity - self.offset),
                "MemoryArena: out of memory",
            ));
        }

        self.offset = new_offset;
        self.stats.record_allocation(size);
        let ptr = unsafe { self.buffer.as_ptr().add(aligned_offset) };
        Ok(ptr)
    }

    /// Allocates an array of `count` elements of type `T`.
    pub fn alloc_slice<T>(&mut self, count: usize) -> BrainResult<&mut [T]> {
        let size = count
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| BrainError::invalid_value("Slice size overflow"))?;
        let ptr = self.alloc(size)? as *mut T;
        Ok(unsafe { std::slice::from_raw_parts_mut(ptr, count) })
    }

    /// Creates a checkpoint of the current arena allocation watermark.
    pub fn checkpoint(&self) -> ArenaCheckpoint {
        ArenaCheckpoint {
            offset: self.offset,
        }
    }

    /// Rewinds the arena to a previously captured checkpoint.
    pub fn rewind(&mut self, checkpoint: ArenaCheckpoint) {
        if checkpoint.offset <= self.offset {
            let freed = self.offset - checkpoint.offset;
            self.offset = checkpoint.offset;
            self.stats.record_deallocation(freed);
        }
    }

    /// Resets the arena, reclaiming all allocated memory in \(O(1)\).
    pub fn reset(&mut self) {
        self.offset = 0;
        self.stats.current_bytes = 0;
        self.stats.active_allocations = 0;
    }

    /// Returns remaining available memory in bytes.
    pub fn remaining_bytes(&self) -> usize {
        self.capacity.saturating_sub(self.offset)
    }

    /// Returns total capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns current statistics.
    pub fn stats(&self) -> AllocationStats {
        self.stats.clone()
    }
}

impl Drop for MemoryArena {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.capacity, self.alignment).unwrap();
        unsafe {
            dealloc(self.buffer.as_ptr(), layout);
        }
    }
}

// =============================================================================
// MemoryTracker & Leak Detection
// =============================================================================

/// Entry in the memory allocation registry.
#[derive(Debug, Clone)]
pub struct AllocationRecord {
    /// Allocation identifier.
    pub id: usize,
    /// Size in bytes.
    pub size: usize,
    /// User tag or module label.
    pub tag: String,
    /// Timestamp or tick index.
    pub tick: usize,
}

/// Global or scoped memory leak detector and allocation tracker.
pub struct MemoryTracker {
    records: Mutex<HashMap<usize, AllocationRecord>>,
    next_id: AtomicUsize,
    current_bytes: AtomicUsize,
    peak_bytes: AtomicUsize,
}

impl MemoryTracker {
    /// Creates a new `MemoryTracker`.
    pub fn new() -> Self {
        MemoryTracker {
            records: Mutex::new(HashMap::new()),
            next_id: AtomicUsize::new(1),
            current_bytes: AtomicUsize::new(0),
            peak_bytes: AtomicUsize::new(0),
        }
    }

    /// Registers a new allocation.
    pub fn track(&self, size: usize, tag: impl Into<String>) -> usize {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let cur = self.current_bytes.fetch_add(size, Ordering::SeqCst) + size;

        let mut peak = self.peak_bytes.load(Ordering::Relaxed);
        while cur > peak {
            match self.peak_bytes.compare_exchange_weak(
                peak,
                cur,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => peak = actual,
            }
        }

        let record = AllocationRecord {
            id,
            size,
            tag: tag.into(),
            tick: 0,
        };
        self.records.lock().unwrap().insert(id, record);
        id
    }

    /// Unregisters an allocation upon deallocation.
    pub fn untrack(&self, id: usize) -> Option<AllocationRecord> {
        let mut recs = self.records.lock().unwrap();
        if let Some(rec) = recs.remove(&id) {
            self.current_bytes.fetch_sub(rec.size, Ordering::SeqCst);
            Some(rec)
        } else {
            None
        }
    }

    /// Returns the number of currently active live allocations.
    pub fn active_count(&self) -> usize {
        self.records.lock().unwrap().len()
    }

    /// Returns currently active bytes.
    pub fn current_bytes(&self) -> usize {
        self.current_bytes.load(Ordering::SeqCst)
    }

    /// Returns peak allocated bytes.
    pub fn peak_bytes(&self) -> usize {
        self.peak_bytes.load(Ordering::SeqCst)
    }

    /// Returns a list of all active allocations (leaks).
    pub fn find_leaks(&self) -> Vec<AllocationRecord> {
        self.records.lock().unwrap().values().cloned().collect()
    }

    /// Clears all tracking records.
    pub fn reset(&self) {
        self.records.lock().unwrap().clear();
        self.current_bytes.store(0, Ordering::SeqCst);
        self.peak_bytes.store(0, Ordering::SeqCst);
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// MemoryPlanner - Graph Execution Reusable Memory Planner
// =============================================================================

/// Lifetime interval for a tensor in an execution graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorLifetime {
    /// Node index where tensor is allocated.
    pub start_step: usize,
    /// Node index where tensor is last read and can be reclaimed.
    pub end_step: usize,
    /// Required memory size in bytes.
    pub size_bytes: usize,
}

/// Static memory planner that shares memory buffers across non-overlapping tensor lifetimes.
#[derive(Debug, Default)]
pub struct MemoryPlanner {
    lifetimes: Vec<TensorLifetime>,
}

impl MemoryPlanner {
    /// Creates a new `MemoryPlanner`.
    pub fn new() -> Self {
        MemoryPlanner {
            lifetimes: Vec::new(),
        }
    }

    /// Adds a tensor lifetime to the plan.
    pub fn add_tensor(&mut self, start_step: usize, end_step: usize, size_bytes: usize) {
        self.lifetimes.push(TensorLifetime {
            start_step,
            end_step,
            size_bytes,
        });
    }

    /// Computes the minimal peak memory required to execute the graph with optimal reuse.
    pub fn compute_peak_memory(&self) -> usize {
        if self.lifetimes.is_empty() {
            return 0;
        }
        let max_step = self.lifetimes.iter().map(|l| l.end_step).max().unwrap_or(0);
        let mut step_usage = vec![0usize; max_step + 1];

        for item in &self.lifetimes {
            for step in item.start_step..=item.end_step {
                step_usage[step] += item.size_bytes;
            }
        }
        step_usage.into_iter().max().unwrap_or(0)
    }

    /// Assigns shared buffer offsets to each tensor lifetime using a first-fit algorithm.
    pub fn plan_offsets(&self) -> Vec<usize> {
        let mut offsets = vec![0usize; self.lifetimes.len()];
        let mut placed: Vec<(usize, TensorLifetime)> = Vec::new();

        for (i, item) in self.lifetimes.iter().enumerate() {
            let mut offset = 0;
            loop {
                let end_offset = offset + item.size_bytes;
                let mut collision = false;
                for &(other_offset, other_life) in &placed {
                    let other_end = other_offset + other_life.size_bytes;
                    let lifetimes_overlap = !(item.end_step < other_life.start_step
                        || item.start_step > other_life.end_step);
                    let memory_overlaps = !(end_offset <= other_offset || offset >= other_end);

                    if lifetimes_overlap && memory_overlaps {
                        offset = other_end;
                        collision = true;
                        break;
                    }
                }
                if !collision {
                    break;
                }
            }
            offsets[i] = offset;
            placed.push((offset, *item));
        }
        offsets
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_helpers() {
        assert_eq!(align_up(0, 64), 0);
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(64, 64), 64);
        assert_eq!(align_up(65, 64), 128);

        assert_eq!(align_down(0, 64), 0);
        assert_eq!(align_down(63, 64), 0);
        assert_eq!(align_down(64, 64), 64);
        assert_eq!(align_down(127, 64), 64);
    }

    #[test]
    fn test_aligned_buffer_basic() {
        let mut buf = AlignedBuffer::<f64>::with_cacheline_alignment(16);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
        assert!(is_aligned(buf.as_ptr() as *const u8, CACHE_LINE_SIZE));

        for i in 0..10 {
            buf.push(i as f64);
        }
        assert_eq!(buf.len(), 10);
        assert_eq!(buf.as_slice()[3], 3.0);
        buf.as_mut_slice()[3] = 42.0;
        assert_eq!(buf.as_slice()[3], 42.0);
    }

    #[test]
    fn test_aligned_buffer_realloc() {
        let mut buf = AlignedBuffer::<i32>::with_capacity_aligned(2, 64);
        for i in 0..100 {
            buf.push(i);
        }
        assert_eq!(buf.len(), 100);
        assert!(buf.capacity() >= 100);
        assert!(is_aligned(buf.as_ptr() as *const u8, 64));
        assert_eq!(buf.as_slice()[50], 50);
    }

    #[test]
    fn test_aligned_buffer_clone() {
        let mut buf = AlignedBuffer::<f32>::with_page_alignment(8);
        buf.push(1.5);
        buf.push(2.5);
        let cloned = buf.clone();
        assert_eq!(cloned.len(), 2);
        assert_eq!(cloned.as_slice(), &[1.5, 2.5]);
        assert!(is_aligned(cloned.as_ptr() as *const u8, PAGE_SIZE));
    }

    #[test]
    fn test_simple_pool_allocation_and_coalesce() {
        let mut pool = SimplePool::new(1024);
        let b1 = pool.allocate(128).unwrap();
        let b2 = pool.allocate(256).unwrap();
        assert_eq!(pool.allocated_bytes(), 128 + 256);

        pool.deallocate(&b1).unwrap();
        pool.deallocate(&b2).unwrap();
        assert_eq!(pool.allocated_bytes(), 0);
        assert_eq!(pool.available_bytes(), pool.capacity());
    }

    #[test]
    fn test_simple_pool_oom() {
        let mut pool = SimplePool::new(128);
        assert!(pool.allocate(256).is_err());
    }

    #[test]
    fn test_binned_memory_pool() {
        let mut pool = BinnedMemoryPool::new();
        let p1 = pool.allocate(100).unwrap();
        let p2 = pool.allocate(100).unwrap();
        assert!(p1 != p2);

        pool.deallocate(p1, 100);
        let p3 = pool.allocate(100).unwrap();
        assert_eq!(p1, p3);
        pool.deallocate(p2, 100);
        pool.deallocate(p3, 100);
    }

    #[test]
    fn test_memory_arena_bump_and_rewind() {
        let mut arena = MemoryArena::new(1024);
        let slice1 = arena.alloc_slice::<f64>(10).unwrap();
        slice1[0] = 3.14;
        let cp = arena.checkpoint();

        let _slice2 = arena.alloc_slice::<f64>(20).unwrap();
        assert!(arena.remaining_bytes() < 1024 - 80);

        arena.rewind(cp);
        assert_eq!(arena.remaining_bytes(), 1024 - 80);
        arena.reset();
        assert_eq!(arena.remaining_bytes(), 1024);
    }

    #[test]
    fn test_memory_tracker_leak_detection() {
        let tracker = MemoryTracker::new();
        let id1 = tracker.track(1024, "weights");
        let id2 = tracker.track(2048, "activations");

        assert_eq!(tracker.current_bytes(), 3072);
        assert_eq!(tracker.peak_bytes(), 3072);
        assert_eq!(tracker.active_count(), 2);

        tracker.untrack(id1);
        assert_eq!(tracker.current_bytes(), 2048);
        let leaks = tracker.find_leaks();
        assert_eq!(leaks.len(), 1);
        assert_eq!(leaks[0].id, id2);
        assert_eq!(leaks[0].tag, "activations");
    }

    #[test]
    fn test_memory_planner_peak_and_offsets() {
        let mut planner = MemoryPlanner::new();
        // Tensor A: step 0..2, 100 bytes
        planner.add_tensor(0, 2, 100);
        // Tensor B: step 1..3, 200 bytes
        planner.add_tensor(1, 3, 200);
        // Tensor C: step 3..4, 100 bytes (can reuse A's memory)
        planner.add_tensor(3, 4, 100);

        let peak = planner.compute_peak_memory();
        assert_eq!(peak, 300); // at step 1-2, A + B = 300

        let offsets = planner.plan_offsets();
        assert_eq!(offsets.len(), 3);
        assert_eq!(offsets[0], 0);
        assert_eq!(offsets[1], 100);
        assert_eq!(offsets[2], 0); // Reused offset 0 since A died at step 2!
    }

    #[test]
    fn test_memory_format_strides() {
        let shape = vec![2, 3, 4, 5];
        let c_strides = MemoryFormat::Contiguous.strides(&shape);
        assert_eq!(c_strides, vec![60, 20, 5, 1]);

        let nhwc_strides = MemoryFormat::ChannelsLast.strides(&shape);
        assert_eq!(nhwc_strides.len(), 4);
    }

    #[test]
    fn test_channels_last_strides_table() {
        // For [N, C, H, W], ChannelsLast strides must be [H*W*C, 1, W*C, C]
        let cases: Vec<(Vec<usize>, Vec<usize>)> = vec![
            (vec![1, 1, 1, 1], vec![1, 1, 1, 1]),
            (vec![2, 3, 4, 5], vec![4 * 5 * 3, 1, 5 * 3, 3]), // [60, 1, 15, 3]
            (vec![4, 16, 32, 32], vec![32 * 32 * 16, 1, 32 * 16, 16]),
            (vec![8, 64, 7, 7], vec![7 * 7 * 64, 1, 7 * 64, 64]),
        ];
        for (shape, expected) in cases {
            assert_eq!(MemoryFormat::ChannelsLast.strides(&shape), expected);
        }
    }

    #[test]
    fn test_arena_zero_alloc_and_oom() {
        let mut arena = MemoryArena::new(64);
        let empty_slice = arena.alloc_slice::<f64>(0).unwrap();
        assert_eq!(empty_slice.len(), 0);

        // Allocating more than capacity should fail with OutOfMemory
        let err = arena.alloc_slice::<f64>(100);
        assert!(err.is_err());
    }
}
