//! Which cores the encoder's worker threads run on.
//!
//! On a big.LITTLE part (the Amlogic S922X has four Cortex-A73 and two
//! Cortex-A53) a worker the scheduler puts on a slow core is the last to
//! finish its share of an interval, and the whole fork waits for it. So the
//! workers are confined to the fast cores, and which those are is, in order:
//!
//! 1. what the host asked for with [`set_cores`] (the C function
//!    `hz_stream_set_cpu_cores`) before the first context was created;
//! 2. the `HZ_STREAM_CORES` environment variable, a comma-separated list
//!    such as `2,3,4,5`, or `none` for no pinning;
//! 3. on Linux, the cores that report the highest capacity (or, where the
//!    kernel gives none, the highest maximum frequency), when not all cores
//!    are alike;
//! 4. nothing: the scheduler decides. This is the answer off Linux.
//!
//! If the cores cannot be told, `taskset -c 2-5` on the player process does
//! the same job from outside.

use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};

static EXPLICIT: AtomicBool = AtomicBool::new(false);
static ONCE: Once = Once::new();

/// Name the cores the worker threads are confined to; an empty list leaves
/// them unconfined. Takes effect only if called before the first context is
/// created, and returns whether it did.
pub fn set_cores(cores: &[usize]) -> bool {
    EXPLICIT.store(true, Ordering::Release);
    hz_mlp::set_pool_cores(cores)
}

/// Apply the default policy once, unless the host already chose.
pub(crate) fn configure() {
    ONCE.call_once(|| {
        if EXPLICIT.load(Ordering::Acquire) {
            return;
        }
        let cores = std::env::var("HZ_STREAM_CORES")
            .ok()
            .and_then(|text| parse_list(&text))
            .unwrap_or_else(detect);
        if !cores.is_empty() && hz_mlp::set_pool_cores(&cores) {
            log::info!("hz-stream: worker threads confined to cores {cores:?}");
        }
    });
}

/// `"2,3,4,5"` is those cores, `"none"` and `""` are no cores; anything else
/// is not a list.
fn parse_list(text: &str) -> Option<Vec<usize>> {
    let text = text.trim();
    if text.is_empty() || text.eq_ignore_ascii_case("none") {
        return Some(Vec::new());
    }
    text.split(',')
        .map(|part| part.trim().parse().ok())
        .collect()
}

/// The indices of the cores whose `value` is within a quarter of the highest
/// (so a lone prime core does not leave the pool one thread), if the cores
/// differ and every one has a value. All alike, any unknown, or fewer than
/// two fast cores means no cores.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn fastest(values: &[Option<u64>]) -> Vec<usize> {
    let Some(known) = values.iter().copied().collect::<Option<Vec<u64>>>() else {
        return Vec::new();
    };
    let top = known.iter().copied().max().unwrap_or(0);
    let picked: Vec<usize> = known
        .iter()
        .enumerate()
        .filter(|(_, v)| **v * 4 >= top * 3)
        .map(|(i, _)| i)
        .collect();
    if picked.len() == known.len() || picked.len() < 2 {
        Vec::new()
    } else {
        picked
    }
}

#[cfg(target_os = "linux")]
fn detect() -> Vec<usize> {
    let read = |cpu: usize, file: &str| -> Option<u64> {
        std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/{file}"))
            .ok()?
            .trim()
            .parse()
            .ok()
    };
    let count = (0..1024)
        .take_while(|cpu| {
            std::path::Path::new(&format!("/sys/devices/system/cpu/cpu{cpu}")).exists()
        })
        .count();
    for file in ["cpu_capacity", "cpufreq/cpuinfo_max_freq"] {
        let values: Vec<Option<u64>> = (0..count).map(|cpu| read(cpu, file)).collect();
        let picked = fastest(&values);
        if !picked.is_empty() {
            return picked;
        }
    }
    Vec::new()
}

#[cfg(not(target_os = "linux"))]
fn detect() -> Vec<usize> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_core_list_is_parsed_strictly() {
        assert_eq!(parse_list("2,3, 4 ,5"), Some(vec![2, 3, 4, 5]));
        assert_eq!(parse_list("none"), Some(vec![]));
        assert_eq!(parse_list(" "), Some(vec![]));
        assert_eq!(parse_list("2,x"), None);
        assert_eq!(parse_list("2-5"), None);
    }

    #[test]
    fn the_fast_cores_are_the_ones_with_the_most() {
        // The S922X: two A53 then four A73.
        let capacity = [
            Some(446),
            Some(446),
            Some(1024),
            Some(1024),
            Some(1024),
            Some(1024),
        ];
        assert_eq!(fastest(&capacity), vec![2, 3, 4, 5]);
        // The cores are alike, or one is unknown: no pinning.
        assert!(fastest(&[Some(1024); 4]).is_empty());
        assert!(fastest(&[Some(1024), None, Some(446)]).is_empty());
        assert!(fastest(&[]).is_empty());
        // A lone prime core is not a pool; a prime and three big are.
        assert!(fastest(&[Some(300), Some(300), Some(1024), Some(500)]).is_empty());
        assert_eq!(
            fastest(&[Some(300), Some(1024), Some(900), Some(850), Some(800)]),
            vec![1, 2, 3, 4]
        );
    }
}
