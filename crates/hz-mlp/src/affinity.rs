//! Where the shared pool's threads may run.
//!
//! A big.LITTLE part gives a pool of equal workers four fast cores and two
//! slow ones, and the scheduler is free to put a worker on a slow one, where
//! the interval it holds is the last to finish and the whole fork waits for
//! it. A caller that knows which cores are fast says so here, before the pool
//! is built, and each worker is then confined to them. Nothing is pinned by
//! default.
//!
//! Dependency-free like the rest of the crate: the one system call is
//! declared here rather than pulled in from a binding crate.

use std::sync::Mutex;

static CORES: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// Set the cores the shared pool's threads are confined to (empty: wherever
/// the scheduler likes, the default). Only has an effect before the pool is
/// built, which is the first encoder with more than one channel; returns
/// whether the setting was taken. A no-op off Linux.
pub fn set_pool_cores(cores: &[usize]) -> bool {
    if !cfg!(target_os = "linux") && !cores.is_empty() {
        return false;
    }
    if crate::encoder::pool_exists() {
        return false;
    }
    *CORES.lock().unwrap_or_else(|e| e.into_inner()) = cores.to_vec();
    true
}

/// Confine the calling thread to the cores set, if any, as the pool's workers
/// are. For a host's own pipeline threads that run next to the pool. A no-op
/// when no cores are set, off Linux, or without the `parallel` feature.
pub fn confine_current_thread() {
    #[cfg(feature = "parallel")]
    {
        let cores = pool_cores();
        if !cores.is_empty() {
            pin_current_thread(&cores);
        }
    }
}

/// The cores set, for the pool to size itself by and pin to.
#[cfg(feature = "parallel")]
pub(crate) fn pool_cores() -> Vec<usize> {
    CORES.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Confine the calling thread to `cores`. Best effort: a refusal (a core
/// that is not there, a container that forbids it) leaves the thread where
/// it was.
#[cfg(all(feature = "parallel", target_os = "linux"))]
pub(crate) fn pin_current_thread(cores: &[usize]) {
    // A cpu_set_t is a bit set of `unsigned long`s, a thousand and twenty-four
    // bits of it. Words of 64 bits hold the same bits on a little-endian
    // machine whatever the width of `unsigned long`.
    const WORDS: usize = 16;
    unsafe extern "C" {
        fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
    }
    let mut mask = [0u64; WORDS];
    for core in cores {
        if *core < WORDS * 64 {
            mask[core / 64] |= 1 << (core % 64);
        }
    }
    // SAFETY: `mask` is a live buffer of the size passed, and pid 0 is the
    // calling thread.
    unsafe {
        sched_setaffinity(0, std::mem::size_of_val(&mask), mask.as_ptr());
    }
}

#[cfg(all(feature = "parallel", not(target_os = "linux")))]
pub(crate) fn pin_current_thread(_cores: &[usize]) {}

#[cfg(all(test, feature = "parallel", target_os = "linux"))]
mod tests {
    use super::pin_current_thread;

    unsafe extern "C" {
        fn sched_getaffinity(pid: i32, size: usize, mask: *mut u64) -> i32;
    }

    fn allowed() -> Vec<usize> {
        let mut mask = [0u64; 16];
        // SAFETY: `mask` is a live buffer of the size passed; pid 0 is the
        // calling thread.
        let rc = unsafe { sched_getaffinity(0, std::mem::size_of_val(&mask), mask.as_mut_ptr()) };
        assert_eq!(rc, 0);
        (0..16 * 64)
            .filter(|core| mask[core / 64] >> (core % 64) & 1 == 1)
            .collect()
    }

    #[test]
    fn a_thread_is_confined_to_the_cores_asked_for_and_to_nothing_else() {
        // On a thread of its own, so the test runner's threads stay put.
        std::thread::spawn(|| {
            let before = allowed();
            let core = before[0];
            pin_current_thread(&[core]);
            assert_eq!(allowed(), vec![core]);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn cores_the_set_cannot_hold_are_ignored() {
        std::thread::spawn(|| {
            let before = allowed();
            // Out of range: nothing set, and the kernel refuses an empty mask.
            pin_current_thread(&[100_000]);
            assert_eq!(allowed(), before);
        })
        .join()
        .unwrap();
    }
}
