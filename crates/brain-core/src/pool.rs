//! # High-Performance Persistent Worker Thread Pool
//!
//! Eliminates per-call `std::thread::scope` overhead for tensor operations, GEMM, and parallel map/reductions.

use std::sync::{Arc, Mutex, Condvar, OnceLock};
use std::thread::{self, JoinHandle};

type Task = Box<dyn FnOnce() + Send + 'static>;

struct PoolInner {
    tasks: Vec<Task>,
    active_count: usize,
    shutdown: bool,
}

/// Persistent Worker Thread Pool.
pub struct ThreadPool {
    inner: Arc<(Mutex<PoolInner>, Condvar)>,
    done_cvar: Arc<Condvar>,
    _workers: Vec<JoinHandle<()>>,
    num_threads: usize,
}

impl ThreadPool {
    /// Creates a new ThreadPool with the specified number of worker threads.
    pub fn new(num_threads: usize) -> Self {
        let num_threads = num_threads.max(1);
        let inner = Arc::new((
            Mutex::new(PoolInner {
                tasks: Vec::new(),
                active_count: 0,
                shutdown: false,
            }),
            Condvar::new(),
        ));
        let done_cvar = Arc::new(Condvar::new());

        let mut workers = Vec::with_capacity(num_threads);
        for _ in 0..num_threads {
            let inner_clone = Arc::clone(&inner);
            let done_clone = Arc::clone(&done_cvar);

            let handle = thread::Builder::new()
                .name("brain-worker".into())
                .spawn(move || {
                    let (lock, cvar) = &*inner_clone;
                    loop {
                        let task = {
                            let mut state = lock.lock().unwrap();
                            while state.tasks.is_empty() && !state.shutdown {
                                state = cvar.wait(state).unwrap();
                            }
                            if state.shutdown && state.tasks.is_empty() {
                                break;
                            }
                            state.tasks.pop()
                        };

                        if let Some(task) = task {
                            task();
                            let mut state = lock.lock().unwrap();
                            state.active_count = state.active_count.saturating_sub(1);
                            if state.active_count == 0 && state.tasks.is_empty() {
                                done_clone.notify_all();
                            }
                        }
                    }
                })
                .expect("Failed to spawn brain-worker thread");
            workers.push(handle);
        }

        Self {
            inner,
            done_cvar,
            _workers: workers,
            num_threads,
        }
    }

    /// Number of worker threads in the pool.
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Executes `f` across index range `0..len` partitioned into chunks.
    /// If `len < min_chunk_size` or `num_threads == 1`, executes sequentially on the calling thread.
    pub fn parallel_for<F>(&self, len: usize, min_chunk_size: usize, f: F)
    where
        F: Fn(usize, usize) + Send + Sync + 'static,
    {
        if len == 0 {
            return;
        }
        if len <= min_chunk_size || self.num_threads <= 1 {
            f(0, len);
            return;
        }

        let num_chunks = self.num_threads.min((len + min_chunk_size - 1) / min_chunk_size);
        if num_chunks <= 1 {
            f(0, len);
            return;
        }

        let chunk_size = (len + num_chunks - 1) / num_chunks;
        let f = Arc::new(f);

        let (lock, cvar) = &*self.inner;
        {
            let mut state = lock.lock().unwrap();
            state.active_count += num_chunks;

            for i in 0..num_chunks {
                let start = i * chunk_size;
                let end = (start + chunk_size).min(len);
                if start >= len {
                    state.active_count -= 1;
                    continue;
                }
                let f_clone = Arc::clone(&f);
                state.tasks.push(Box::new(move || {
                    f_clone(start, end);
                }));
            }
            cvar.notify_all();
        }

        // Wait for completion
        let mut state = lock.lock().unwrap();
        while state.active_count > 0 || !state.tasks.is_empty() {
            state = self.done_cvar.wait(state).unwrap();
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        let (lock, cvar) = &*self.inner;
        {
            let mut state = lock.lock().unwrap();
            state.shutdown = true;
            cvar.notify_all();
        }
    }
}

static GLOBAL_POOL: OnceLock<ThreadPool> = OnceLock::new();

/// Returns a reference to the global shared worker thread pool.
pub fn global_pool() -> &'static ThreadPool {
    GLOBAL_POOL.get_or_init(|| {
        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(2)
            .min(4); // Respect concurrency bounds
        ThreadPool::new(num_threads)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_thread_pool_parallel_for() {
        let pool = ThreadPool::new(4);
        let counter = Arc::new(AtomicUsize::new(0));

        let counter_clone = Arc::clone(&counter);
        pool.parallel_for(100, 10, move |start, end| {
            counter_clone.fetch_add(end - start, Ordering::SeqCst);
        });

        assert_eq!(counter.load(Ordering::SeqCst), 100);
    }
}
