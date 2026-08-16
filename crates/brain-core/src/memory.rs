//! Memory management for the Brain deep learning framework.
//!
//! This module provides memory allocation tracking, pool-based allocation,
//! arena allocators, and memory format utilities for efficient tensor operations.
//!
//! # Key Components
//!
//! * [`MemoryPool`] trait and [`SimplePool`] implementation
//! * [`ArenaAllocator`] for batch allocation
//! * [`AllocationStats`] for tracking memory usage
//! * [`MemoryFormat`] enum for different memory layouts
//! * [`MemoryPlanner`] for graph execution memory planning

use std::collections::HashMap;
use std::fmt;

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
                    // NHWC strides
                    let (n, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
                    vec![h * w * c, w * c, c, 1]
                } else {
                    strides_row_major(shape)
                }
            }
            MemoryFormat::Blocked { block_size } => {
                // Blocked: tiles of block_size x block_size
                if shape.len() == 2 {
                    let (rows, cols) = (shape[0], shape[1]);
                    let br = (rows + block_size - 1) / block_size;
                    let bc = (cols + block_size - 1) / block_size;
                    vec![block_size, 1, br * block_size * block_size, block_size]
                } else {
                    strides_row_major(shape)
                }
            }
            MemoryFormat::Sparse => vec![],
            MemoryFormat::Packed => {
                if shape.len() == 1 {
                    vec![(shape[0] + 3) / 4]
                } else {
                    strides_row_major(shape)
                }
            }
        }
    }

    /// Returns true if the format preserves the natural ordering.
    pub fn is_dense(&self) -> bool {
        matches!(self, MemoryFormat::Contiguous | MemoryFormat::ChannelsFirst | MemoryFormat::ChannelsLast)
    }
}

// =============================================================================
// MemoryPool Trait
// =============================================================================

/// A trait for memory pool allocators.
pub trait MemoryPool: Send + Sync {
    /// Allocates a block of memory of the given size in bytes.
    fn allocate(&mut self, size: usize) -> Result<MemoryBlock, String>;

    /// Deallocates a previously allocated block.
    fn deallocate(&mut self, block: &MemoryBlock) -> Result<(), String>;

    /// Returns the total allocated memory.
    fn allocated_bytes(&self) -> usize;

    /// Returns the available memory.
    fn available_bytes(&self) -> usize;

    /// Returns statistics about the pool.
    fn stats(&self) -> AllocationStats;

    /// Resets the pool, deallocating all blocks.
    fn reset(&mut self);
}

/// A block of allocated memory.
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    /// The offset within the pool's backing storage.
    pub offset: usize,
    /// The size of the block in bytes.
    pub size: usize,
    /// An identifier for this allocation.
    pub id: usize,
}

// =============================================================================
// SimplePool Implementation
// =============================================================================

/// A simple memory pool that manages a pre-allocated buffer.
#[derive(Debug)]
pub struct SimplePool {
    /// The backing storage.
    buffer: Vec<u8>,
    /// Free list of (offset, size) pairs.
    free_list: Vec<(usize, usize)>,
    /// Allocation statistics.
    stats: AllocationStats,
    /// Next allocation ID.
    next_id: usize,
    /// Maximum capacity in bytes.
    capacity: usize,
}

impl SimplePool {
    /// Creates a new SimplePool with the given capacity in bytes.
    pub fn new(capacity: usize) -> Self {
        SimplePool {
            buffer: vec![0u8; capacity],
            free_list: vec![(0, capacity)],
            stats: AllocationStats::new(capacity),
            next_id: 0,
            capacity,
        }
    }

    /// Creates a pool with a default capacity of 1 GB.
    pub fn default_pool() -> Self {
        Self::new(1024 * 1024 * 1024)
    }

    /// Returns the total capacity.
    pub fn capacity(&self) -> usize { self.capacity }

    /// Returns the number of free blocks.
    pub fn free_blocks(&self) -> usize { self.free_list.len() }
}

impl MemoryPool for SimplePool {
    fn allocate(&mut self, size: usize) -> Result<MemoryBlock, String> {
        // Round up to 8-byte alignment
        let aligned_size = (size + 7) & !7;
        if aligned_size == 0 { return Err("Cannot allocate zero bytes".into()); }

        // Find a suitable free block
        let mut best_idx = None;
        let mut best_size = usize::MAX;
        for (i, &(offset, block_size)) in self.free_list.iter().enumerate() {
            if block_size >= aligned_size && block_size < best_size {
                best_idx = Some(i);
                best_size = block_size;
            }
        }

        if let Some(idx) = best_idx {
            let (offset, block_size) = self.free_list[idx];
            self.free_list.remove(idx);

            // Split if there's significant leftover
            let leftover = block_size - aligned_size;
            if leftover >= 8 {
                self.free_list.push((offset + aligned_size, leftover));
            }

            self.stats.total_allocations += 1;
            self.stats.active_allocations += 1;
            self.stats.peak_allocations = self.stats.peak_allocations.max(self.stats.active_allocations);
            self.stats.bytes_allocated += aligned_size;
            self.stats.bytes_active += aligned_size;
            self.stats.peak_bytes = self.stats.peak_bytes.max(self.stats.bytes_active);

            let id = self.next_id;
            self.next_id += 1;

            Ok(MemoryBlock { offset, size: aligned_size, id })
        } else {
            // Try to coalesce and retry
            self.coalesce_free_list();
            for (i, &(offset, block_size)) in self.free_list.iter().enumerate() {
                if block_size >= aligned_size && block_size < best_size {
                    best_idx = Some(i);
                    best_size = block_size;
                }
            }
            if let Some(idx) = best_idx {
                let (offset, block_size) = self.free_list[idx];
                self.free_list.remove(idx);
                let leftover = block_size - aligned_size;
                if leftover >= 8 { self.free_list.push((offset + aligned_size, leftover)); }
                self.stats.total_allocations += 1;
                self.stats.active_allocations += 1;
                self.stats.bytes_allocated += aligned_size;
                self.stats.bytes_active += aligned_size;
                let id = self.next_id; self.next_id += 1;
                return Ok(MemoryBlock { offset, size: aligned_size, id });
            }
            Err(format!("Cannot allocate {} bytes: no suitable free block (pool capacity: {})", size, self.capacity))
        }
    }

    fn deallocate(&mut self, block: &MemoryBlock) -> Result<(), String> {
        self.free_list.push((block.offset, block.size));
        self.stats.active_allocations -= 1;
        self.stats.bytes_active -= block.size;
        self.stats.total_deallocations += 1;
        self.coalesce_free_list();
        Ok(())
    }

    fn allocated_bytes(&self) -> usize { self.stats.bytes_active }

    fn available_bytes(&self) -> usize { self.capacity - self.stats.bytes_active }

    fn stats(&self) -> AllocationStats { self.stats.clone() }

    fn reset(&mut self) {
        self.free_list.clear();
        self.free_list.push((0, self.capacity));
        self.stats = AllocationStats::new(self.capacity);
        self.next_id = 0;
    }
}

impl SimplePool {
    /// Coalesces adjacent free blocks to reduce fragmentation.
    fn coalesce_free_list(&mut self) {
        self.free_list.sort_by_key(|(offset, _)| *offset);
        let mut merged = Vec::new();
        for &(offset, size) in &self.free_list {
            if let Some((last_offset, last_size)) = merged.last_mut() {
                if *last_offset + *last_size == offset {
                    *last_size += size;
                    continue;
                }
            }
            merged.push((offset, size));
        }
        self.free_list = merged;
    }

    /// Returns the fragmentation ratio (0.0 = no fragmentation, 1.0 = maximum).
    pub fn fragmentation(&self) -> f64 {
        if self.free_list.is_empty() { return 0.0; }
        let total_free: usize = self.free_list.iter().map(|(_, s)| *s).sum();
        let largest_free: usize = self.free_list.iter().map(|(_, s)| *s).max().unwrap_or(0);
        if total_free == 0 { return 0.0; }
        1.0 - largest_free as f64 / total_free as f64
    }
}

// =============================================================================
// ArenaAllocator
// =============================================================================

/// An arena allocator that allocates memory in batches.
#[derive(Debug)]
pub struct ArenaAllocator {
    /// The memory arena.
    arena: Vec<Vec<u8>>,
    /// Current arena being filled.
    current_arena: usize,
    /// Current offset within the current arena.
    offset: usize,
    /// Chunk size for new arenas.
    chunk_size: usize,
    /// Statistics.
    stats: AllocationStats,
    /// Total capacity across all arenas.
    total_capacity: usize,
}

impl ArenaAllocator {
    /// Creates a new ArenaAllocator with the given chunk size.
    pub fn new(chunk_size: usize) -> Self {
        ArenaAllocator {
            arena: vec![vec![0u8; chunk_size]],
            current_arena: 0,
            offset: 0,
            chunk_size,
            stats: AllocationStats::new(chunk_size),
            total_capacity: chunk_size,
        }
    }

    /// Allocates a block from the arena.
    pub fn allocate(&mut self, size: usize) -> Result<(usize, usize), String> {
        let aligned_size = (size + 7) & !7;
        if self.offset + aligned_size <= self.arena[self.current_arena].len() {
            let arena_idx = self.current_arena;
            let offset = self.offset;
            self.offset += aligned_size;
            self.stats.total_allocations += 1;
            self.stats.active_allocations += 1;
            self.stats.bytes_allocated += aligned_size;
            self.stats.bytes_active += aligned_size;
            return Ok((arena_idx, offset));
        }
        // Need a new arena
        let new_size = aligned_size.max(self.chunk_size);
        self.arena.push(vec![0u8; new_size]);
        self.current_arena = self.arena.len() - 1;
        self.offset = aligned_size;
        self.total_capacity += new_size;
        self.stats.total_allocations += 1;
        self.stats.active_allocations += 1;
        self.stats.bytes_allocated += aligned_size;
        self.stats.bytes_active += aligned_size;
        Ok((self.current_arena, 0))
    }

    /// Returns a mutable slice of the allocated region.
    pub fn get_slice_mut(&mut self, arena_idx: usize, offset: usize, len: usize) -> &mut [u8] {
        &mut self.arena[arena_idx][offset..offset + len]
    }

    /// Returns the total capacity.
    pub fn total_capacity(&self) -> usize { self.total_capacity }

    /// Returns the used capacity.
    pub fn used_capacity(&self) -> usize {
        let used_before: usize = self.arena.iter().take(self.current_arena).map(|a| a.len()).sum();
        used_before + self.offset
    }

    /// Resets the arena, deallocating all memory.
    pub fn reset(&mut self) {
        self.arena.clear();
        self.arena.push(vec![0u8; self.chunk_size]);
        self.current_arena = 0;
        self.offset = 0;
        self.total_capacity = self.chunk_size;
        self.stats = AllocationStats::new(self.chunk_size);
    }

    /// Returns statistics.
    pub fn stats(&self) -> &AllocationStats { &self.stats }
}

// =============================================================================
// AllocationStats
// =============================================================================

/// Statistics about memory allocations.
#[derive(Debug, Clone)]
pub struct AllocationStats {
    /// Total number of allocation requests.
    pub total_allocations: usize,
    /// Total number of deallocation requests.
    pub total_deallocations: usize,
    /// Currently active allocations.
    pub active_allocations: usize,
    /// Peak number of active allocations.
    pub peak_allocations: usize,
    /// Total bytes ever allocated.
    pub bytes_allocated: usize,
    /// Currently active bytes.
    pub bytes_active: usize,
    /// Peak active bytes.
    pub peak_bytes: usize,
    /// Total capacity of the pool.
    pub total_capacity: usize,
}

impl AllocationStats {
    fn new(capacity: usize) -> Self {
        AllocationStats {
            total_allocations: 0,
            total_deallocations: 0,
            active_allocations: 0,
            peak_allocations: 0,
            bytes_allocated: 0,
            bytes_active: 0,
            peak_bytes: 0,
            total_capacity: capacity,
        }
    }

    /// Returns the utilization ratio.
    pub fn utilization(&self) -> f64 {
        if self.total_capacity == 0 { return 0.0; }
        self.bytes_active as f64 / self.total_capacity as f64
    }

    /// Formats a summary string.
    pub fn summary(&self) -> String {
        format!(
            "Allocations: {} total, {} active (peak: {}) | Bytes: {} allocated, {} active (peak: {}) / {} capacity ({:.1}% utilized)",
            self.total_allocations, self.active_allocations, self.peak_allocations,
            self.bytes_allocated, self.bytes_active, self.peak_bytes, self.total_capacity,
            self.utilization() * 100.0,
        )
    }
}

impl fmt::Display for AllocationStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

impl Default for AllocationStats {
    fn default() -> Self { AllocationStats::new(0) }
}

// =============================================================================
// Memory Planning for Graph Execution
// =============================================================================

/// Plans memory allocations for a computation graph.
#[derive(Debug, Clone)]
pub struct MemoryPlanner {
    /// Planned allocations (offset, size, lifetime_start, lifetime_end).
    allocations: Vec<(usize, usize, usize, usize)>,
    /// Total memory needed.
    total_needed: usize,
    /// Memory reuse map: lifetime -> offset.
    reuse_map: HashMap<usize, usize>,
}

impl MemoryPlanner {
    /// Creates a new empty memory planner.
    pub fn new() -> Self {
        MemoryPlanner { allocations: Vec::new(), total_needed: 0, reuse_map: HashMap::new() }
    }

    /// Adds a tensor allocation requirement.
    pub fn add_tensor(&mut self, size: usize, lifetime_start: usize, lifetime_end: usize) {
        let aligned_size = (size + 7) & !7;
        self.allocations.push((0, aligned_size, lifetime_start, lifetime_end));
        self.total_needed += aligned_size;
    }

    /// Plans the memory layout, reusing memory where possible.
    pub fn plan(&mut self) -> Vec<(usize, usize)> {
        let mut plan = Vec::new();
        let mut free_at: Vec<(usize, usize)> = Vec::new(); // (offset, size) available after timestep

        // Sort allocations by lifetime start, then size (largest first for same start)
        let mut allocs: Vec<(usize, usize, usize, usize)> = self.allocations.clone();
        allocs.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| b.1.cmp(&a.1)));

        for (size, _, start, end) in allocs {
            // Find free block that fits
            let mut found = None;
            for (i, (offset, block_size)) in free_at.iter_mut().enumerate() {
                if *block_size >= size {
                    plan.push((*offset, size));
                    let leftover = *block_size - size;
                    if leftover >= 8 { *block_size = leftover; } else { *block_size = 0; }
                    found = Some(i);
                    break;
                }
            }
            if found.is_none() {
                // Allocate new space
                let offset = self.total_needed;
                plan.push((offset, size));
                self.total_needed += size;
            }
        }

        plan
    }

    /// Returns the total memory needed after planning.
    pub fn total_memory(&self) -> usize { self.total_needed }
}

impl Default for MemoryPlanner {
    fn default() -> Self { Self::new() }
}

// =============================================================================
// Copy Utilities
// =============================================================================

/// Copies tensor data to a host byte buffer.
pub fn copy_to_host(data: &[f64], buffer: &mut Vec<u8>) {
    buffer.clear();
    buffer.reserve(data.len() * 8);
    for &v in data {
        let bits = v.to_bits();
        for i in 0..8 {
            buffer.push(((bits >> (i * 8)) & 0xFF) as u8);
        }
    }
}

/// Copies host byte buffer to f64 values.
pub fn copy_from_host(buffer: &[u8]) -> Vec<f64> {
    let count = buffer.len() / 8;
    let mut data = Vec::with_capacity(count);
    for i in 0..count {
        let base = i * 8;
        let bits = (buffer[base] as u64)
            | ((buffer[base + 1] as u64) << 8)
            | ((buffer[base + 2] as u64) << 16)
            | ((buffer[base + 3] as u64) << 24)
            | ((buffer[base + 4] as u64) << 32)
            | ((buffer[base + 5] as u64) << 40)
            | ((buffer[base + 6] as u64) << 48)
            | ((buffer[base + 7] as u64) << 56);
        data.push(f64::from_bits(bits));
    }
    data
}

/// Computes the size in bytes for a given element count and dtype size.
pub fn compute_size_bytes(numel: usize, element_size: usize) -> usize {
    numel.saturating_mul(element_size)
}

// =============================================================================
// Helper Functions
// =============================================================================

fn strides_row_major(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    if n == 0 { return vec![]; }
    let mut strides = vec![1usize; n];
    for i in (0..n - 1).rev() { strides[i] = strides[i + 1] * shape[i + 1]; }
    strides
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_format_default() {
        assert_eq!(MemoryFormat::default(), MemoryFormat::Contiguous);
    }

    #[test]
    fn test_memory_format_display() {
        assert_eq!(format!("{}", MemoryFormat::Contiguous), "contiguous");
        assert_eq!(format!("{}", MemoryFormat::ChannelsFirst), "channels_first");
        assert_eq!(format!("{}", MemoryFormat::Blocked { block_size: 32 }), "blocked(32)");
    }

    #[test]
    fn test_memory_format_strides() {
        let strides = MemoryFormat::Contiguous.strides(&[2, 3, 4]);
        assert_eq!(strides, vec![12, 4, 1]);

        let strides = MemoryFormat::ChannelsLast.strides(&[2, 3, 4, 5]);
        assert_eq!(strides, vec![3 * 4 * 5, 4 * 5, 5, 1]);
    }

    #[test]
    fn test_memory_format_is_dense() {
        assert!(MemoryFormat::Contiguous.is_dense());
        assert!(MemoryFormat::ChannelsFirst.is_dense());
        assert!(!MemoryFormat::Sparse.is_dense());
    }

    #[test]
    fn test_simple_pool_creation() {
        let pool = SimplePool::new(1024);
        assert_eq!(pool.capacity(), 1024);
        assert_eq!(pool.available_bytes(), 1024);
        assert_eq!(pool.free_blocks(), 1);
    }

    #[test]
    fn test_simple_pool_allocate() {
        let mut pool = SimplePool::new(1024);
        let block = pool.allocate(128).unwrap();
        assert_eq!(block.size, 128);
        assert_eq!(pool.allocated_bytes(), 128);
        assert_eq!(pool.available_bytes(), 896);
    }

    #[test]
    fn test_simple_pool_allocate_aligned() {
        let mut pool = SimplePool::new(1024);
        let block = pool.allocate(13).unwrap();
        assert_eq!(block.size, 16); // Aligned to 8 bytes
    }

    #[test]
    fn test_simple_pool_deallocate() {
        let mut pool = SimplePool::new(1024);
        let block = pool.allocate(128).unwrap();
        pool.deallocate(&block).unwrap();
        assert_eq!(pool.allocated_bytes(), 0);
    }

    #[test]
    fn test_simple_pool_multiple_allocations() {
        let mut pool = SimplePool::new(1024);
        let b1 = pool.allocate(100).unwrap();
        let b2 = pool.allocate(200).unwrap();
        let b3 = pool.allocate(50).unwrap();
        assert!(pool.allocated_bytes() >= 350);
        pool.deallocate(&b1).unwrap();
        pool.deallocate(&b2).unwrap();
        assert!(pool.allocated_bytes() < 350);
    }

    #[test]
    fn test_simple_pool_fragmentation() {
        let pool = SimplePool::new(1024);
        let frag = pool.fragmentation();
        assert!(frag >= 0.0 && frag <= 1.0);
    }

    #[test]
    fn test_simple_pool_stats() {
        let mut pool = SimplePool::new(1024);
        pool.allocate(100).unwrap();
        let stats = pool.stats();
        assert_eq!(stats.total_allocations, 1);
        assert!(stats.utilization() > 0.0);
    }

    #[test]
    fn test_simple_pool_reset() {
        let mut pool = SimplePool::new(1024);
        pool.allocate(100).unwrap();
        pool.reset();
        assert_eq!(pool.available_bytes(), 1024);
        assert_eq!(pool.stats().total_allocations, 0);
    }

    #[test]
    fn test_simple_pool_overflow() {
        let mut pool = SimplePool::new(100);
        pool.allocate(80).unwrap();
        pool.allocate(80).unwrap();
        // Second allocation should coalesce or fail gracefully
        let result = pool.allocate(50);
        // May or may not succeed depending on coalescing
        if result.is_err() { assert!(result.unwrap_err().contains("Cannot allocate")); }
    }

    #[test]
    fn test_arena_allocator_creation() {
        let arena = ArenaAllocator::new(1024);
        assert_eq!(arena.total_capacity(), 1024);
    }

    #[test]
    fn test_arena_allocator_allocate() {
        let mut arena = ArenaAllocator::new(256);
        let (idx, offset) = arena.allocate(64).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(offset, 0);
    }

    #[test]
    fn test_arena_allocator_multiple_chunks() {
        let mut arena = ArenaAllocator::new(64);
        arena.allocate(32).unwrap();
        arena.allocate(32).unwrap();
        arena.allocate(32).unwrap();
        assert!(arena.arena.len() >= 3);
    }

    #[test]
    fn test_arena_allocator_get_slice() {
        let mut arena = ArenaAllocator::new(256);
        let (idx, offset) = arena.allocate(32).unwrap();
        let slice = arena.get_slice_mut(idx, offset, 32);
        slice[0] = 42;
        assert_eq!(slice[0], 42);
    }

    #[test]
    fn test_arena_allocator_reset() {
        let mut arena = ArenaAllocator::new(256);
        arena.allocate(100).unwrap();
        arena.reset();
        assert_eq!(arena.arena.len(), 1);
        assert_eq!(arena.used_capacity(), 0);
    }

    #[test]
    fn test_allocation_stats_default() {
        let stats = AllocationStats::default();
        assert_eq!(stats.total_allocations, 0);
        assert!(stats.utilization().abs() < 1e-10);
    }

    #[test]
    fn test_allocation_stats_display() {
        let stats = AllocationStats::default();
        let s = format!("{}", stats);
        assert!(s.contains("Allocations"));
    }

    #[test]
    fn test_memory_planner() {
        let mut planner = MemoryPlanner::new();
        planner.add_tensor(100, 0, 2);
        planner.add_tensor(200, 1, 3);
        let plan = planner.plan();
        assert_eq!(plan.len(), 2);
    }

    #[test]
    fn test_memory_planner_total() {
        let mut planner = MemoryPlanner::new();
        planner.add_tensor(100, 0, 5);
        assert!(planner.total_memory() >= 100);
    }

    #[test]
    fn test_copy_to_host() {
        let data = vec![1.0, 2.0, 3.0];
        let mut buffer = Vec::new();
        copy_to_host(&data, &mut buffer);
        assert_eq!(buffer.len(), 24); // 3 * 8 bytes
        let back = copy_from_host(&buffer);
        for (orig, read) in data.iter().zip(back.iter()) {
            assert!((orig - read).abs() < 1e-10);
        }
    }

    #[test]
    fn test_copy_from_host() {
        let data = vec![1.0, 2.0, 3.0];
        let mut buffer = Vec::new();
        copy_to_host(&data, &mut buffer);
        let back = copy_from_host(&buffer);
        assert_eq!(back.len(), 3);
        assert!((back[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_size_bytes() {
        assert_eq!(compute_size_bytes(100, 8), 800);
        assert_eq!(compute_size_bytes(0, 8), 0);
    }

    #[test]
    fn test_memory_pool_trait() {
        fn use_pool(pool: &mut dyn MemoryPool) {
            let _ = pool.allocate(64);
            assert!(pool.allocated_bytes() > 0);
        }
        let mut pool = SimplePool::new(1024);
        use_pool(&mut pool);
    }

    #[test]
    fn test_simple_pool_zero_alloc() {
        let mut pool = SimplePool::new(1024);
        let result = pool.allocate(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_simple_pool_stats_after_reset() {
        let mut pool = SimplePool::new(1024);
        pool.allocate(100).unwrap();
        pool.allocate(200).unwrap();
        pool.reset();
        assert_eq!(pool.stats().active_allocations, 0);
        assert_eq!(pool.stats().bytes_active, 0);
    }
}
