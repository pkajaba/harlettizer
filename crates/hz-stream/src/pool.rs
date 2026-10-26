//! Buffers that go round the pipeline instead of being allocated per frame.
//!
//! A frame's samples are written by the decode thread, read by the encode
//! thread and then not needed; a packet's bytes are written by the caller,
//! read by the decode thread and then not needed. Each buffer therefore makes
//! a loop (take, fill, send, use, give back) and the loop needs a handful of
//! them for the life of the stream.
//!
//! **Why this is sound.** A buffer is a `Vec` owned by value by exactly one
//! party at any time: the pool while it is idle, the stage that took it while
//! it is being filled, the channel while it is in flight, the stage that
//! received it while it is being read. Sending it transfers ownership, so the
//! sender cannot touch it again and the borrow checker enforces that; nothing
//! here borrows a buffer across a send, and there is no `unsafe`. The pool
//! holds idle buffers only, behind a `Mutex` that is held for a push or a pop
//! and never across anything else. A buffer lost to a closed channel or a
//! panic is dropped with it, and the pool refills by allocating once more.

use std::sync::{Mutex, MutexGuard};

/// Idle buffers of one kind.
pub(crate) struct Pool<T> {
    idle: Mutex<Vec<Vec<T>>>,
    /// More than this many idle buffers are dropped instead of kept, so a
    /// burst does not leave its high-water mark behind for good.
    keep: usize,
    /// Buffers handed out that were not recycled ones.
    #[cfg(test)]
    fresh: std::sync::atomic::AtomicUsize,
}

impl<T> Pool<T> {
    pub(crate) const fn new(keep: usize) -> Self {
        Self {
            idle: Mutex::new(Vec::new()),
            keep,
            #[cfg(test)]
            fresh: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn idle(&self) -> MutexGuard<'_, Vec<Vec<T>>> {
        self.idle.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// An empty buffer, with the capacity of an earlier one when there is one.
    pub(crate) fn take(&self) -> Vec<T> {
        let recycled = self.idle().pop();
        #[cfg(test)]
        if recycled.is_none() {
            self.fresh
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        recycled.unwrap_or_default()
    }

    /// Give a buffer back; it is emptied here.
    pub(crate) fn give(&self, mut buffer: Vec<T>) {
        if buffer.capacity() == 0 {
            return;
        }
        buffer.clear();
        let mut idle = self.idle();
        if idle.len() < self.keep {
            idle.push(buffer);
        }
    }

    /// How many times `take` had nothing to hand back.
    #[cfg(test)]
    pub(crate) fn fresh_count(&self) -> usize {
        self.fresh.load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn idle_count(&self) -> usize {
        self.idle().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_buffer_comes_back_empty_with_its_room() {
        let pool = Pool::<i32>::new(2);
        let mut a = pool.take();
        a.extend(0..1000);
        let room = a.capacity();
        pool.give(a);
        let b = pool.take();
        assert!(b.is_empty());
        assert!(b.capacity() >= room);
    }

    #[test]
    fn only_so_many_are_kept() {
        let pool = Pool::<u8>::new(2);
        for _ in 0..5 {
            pool.give(vec![1, 2, 3]);
        }
        assert_eq!(pool.idle_count(), 2);
        // A buffer that never held anything is not worth keeping.
        pool.give(Vec::new());
        assert_eq!(pool.idle_count(), 2);
    }
}
