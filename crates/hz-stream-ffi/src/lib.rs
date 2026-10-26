//! The C ABI of [`hz_stream`]: a library a player loads at run time.
//!
//! Every entry point returns a negative number instead of unwinding into the
//! host: -1 for a bad argument, -2 for a buffer that is too small, -3 for a
//! stream at a rate the context was not built for, -4 for a pipeline that
//! stopped (a panic caught in the call), -5 for a key that is not clean
//! hexadecimal. The header, `include/hz_stream.h`, is the contract; the
//! constant [`HZ_STREAM_ABI_VERSION`] is bumped whenever it changes and a
//! host compares it with the one it was built against before it calls
//! anything else.
//!
//! **Keys.** The Evolution frames the stream carries have a protection field,
//! a keyed digest. This library ships no key, derives none and helps nobody
//! obtain one (see `LEGAL.md`, section 4). A host may hand it a key of its
//! own through [`hz_stream_set_key`] or [`hz_stream_set_key_hex`], which is
//! forwarded to the encoder's `set_evolution_key` and used for nothing else;
//! without one the field holds a constant that verifies against nothing, and
//! a receiver that checks it may refuse the stream.

use hz_stream::{Context, cores};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::slice;

/// Bumped whenever `include/hz_stream.h` changes in a way a host built against
/// the old one would not survive.
pub const HZ_STREAM_ABI_VERSION: u32 = 1;

/// The ABI version this library implements.
#[unsafe(no_mangle)]
pub extern "C" fn hz_stream_abi_version() -> u32 {
    HZ_STREAM_ABI_VERSION
}

/// The text a failed push carries when the pipeline has stopped.
const STOPPED: &str = hz_stream::STOPPED;

fn guarded(f: impl FnOnce() -> i32) -> i32 {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(-4)
}

/// A new context for a stream at `sample_rate`, or null when the encoder cannot
/// be built for it. `effort_level` is 0 for the real-time encoder (the one a
/// player wants), 1 for fast and anything higher for full.
#[unsafe(no_mangle)]
pub extern "C" fn hz_stream_create(sample_rate: u32, effort_level: u32) -> *mut Context {
    match catch_unwind(|| Context::new(sample_rate, effort_level)) {
        Ok(Ok(ctx)) => Box::into_raw(Box::new(ctx)),
        _ => std::ptr::null_mut(),
    }
}

/// Confine the encoder's worker threads to these cores (`count` core
/// numbers; zero for none). Must be called before the first context is
/// created to have any effect; returns 0 if it did, 1 if it was too late or
/// unsupported here, -1 for a bad argument.
///
/// # Safety
/// `cores` is null with `count` zero, or points at `count` readable `i32`s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hz_stream_set_cpu_cores(cores: *const i32, count: usize) -> i32 {
    if count > 0 && cores.is_null() {
        return -1;
    }
    guarded(|| {
        let list: Vec<usize> = if count == 0 {
            Vec::new()
        } else {
            unsafe { slice::from_raw_parts(cores, count) }
                .iter()
                .filter(|c| **c >= 0)
                .map(|c| *c as usize)
                .collect()
        };
        if cores::set_cores(&list) { 0 } else { 1 }
    })
}

/// # Safety
/// `ctx` is null or came from [`hz_stream_create`] and has not been destroyed; no other call on it is running.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hz_stream_destroy(ctx: *mut Context) {
    if !ctx.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe { drop(Box::from_raw(ctx)) }));
    }
}

/// # Safety
/// `ctx` is null or a live context from [`hz_stream_create`]; no other call on it is running.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hz_stream_reset(ctx: *mut Context) {
    guarded(|| {
        if let Some(ctx) = unsafe { ctx.as_mut() } {
            ctx.reset();
        }
        0
    });
}

/// # Safety
/// `ctx` is null or a live context, and `key_ptr` points at `key_len` readable bytes; no other call on `ctx` is running.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hz_stream_set_key(
    ctx: *mut Context,
    key_ptr: *const u8,
    key_len: usize,
) -> i32 {
    if ctx.is_null() || key_ptr.is_null() || key_len == 0 {
        return -1;
    }
    guarded(|| {
        let key = unsafe { slice::from_raw_parts(key_ptr, key_len) };
        unsafe { &mut *ctx }.set_key(key);
        0
    })
}

/// Set the key from hexadecimal text (`text_ptr`, `text_len` bytes, no NUL
/// needed). Returns 0, -1 for a null argument, -5 if the text is not a clean
/// even-length hex string of 1 to 128 bytes; the key is then unchanged.
///
/// # Safety
/// `ctx` is null or a live context, and `text_ptr` points at `text_len` readable bytes; no other call on `ctx` is running.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hz_stream_set_key_hex(
    ctx: *mut Context,
    text_ptr: *const u8,
    text_len: usize,
) -> i32 {
    if ctx.is_null() || text_ptr.is_null() {
        return -1;
    }
    guarded(|| {
        let bytes = unsafe { slice::from_raw_parts(text_ptr, text_len) };
        let Ok(text) = std::str::from_utf8(bytes) else {
            return -5;
        };
        match unsafe { &mut *ctx }.set_key_hex(text) {
            Ok(()) => 0,
            Err(_) => -5,
        }
    })
}

/// # Safety
/// `ctx` is null or a live context, and `data` points at `len` readable bytes; no other call on `ctx` is running.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hz_stream_push_dts(
    ctx: *mut Context,
    data: *const u8,
    len: usize,
    pts: f64,
) -> i32 {
    if ctx.is_null() || data.is_null() || len == 0 || len > i32::MAX as usize {
        return -1;
    }
    guarded(|| {
        let slice = unsafe { slice::from_raw_parts(data, len) };
        match unsafe { &mut *ctx }.push_dts(slice, pts) {
            Ok(consumed) => consumed as i32,
            Err(e) if e == STOPPED => -4,
            Err(_) => -3,
        }
    })
}

/// # Safety
/// `ctx` is null or a live context, `out_buf` points at `max_len` writable bytes, and `out_pts` is null or points at a writable `f64`; no other call on `ctx` is running.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hz_stream_pull_truehd(
    ctx: *mut Context,
    out_buf: *mut u8,
    max_len: usize,
    out_pts: *mut f64,
) -> i32 {
    if ctx.is_null() || out_buf.is_null() || max_len == 0 {
        return -1;
    }
    guarded(|| {
        let slice = unsafe { slice::from_raw_parts_mut(out_buf, max_len) };
        let mut pts = f64::NAN;
        let res = unsafe { &mut *ctx }.pull_truehd(slice, &mut pts);
        if !out_pts.is_null() {
            unsafe { *out_pts = pts };
        }
        match res {
            Ok(n) => n as i32,
            Err(_) => -2,
        }
    })
}

/// # Safety
/// `ctx` is null or a live context; no other call on it is running.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hz_stream_flush(ctx: *mut Context) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    guarded(|| unsafe { &mut *ctx }.flush().min(i32::MAX as usize) as i32)
}
