//! Pipeline timing on synthetic sixteen-channel input: what a push costs the
//! caller, and how fast the whole pipeline runs flat out. Needs no material.
//!
//! `cargo bench -p hz-stream --bench synthetic`

use std::time::Instant;

use hz_stream::Context;

struct Stats {
    count: usize,
    min_us: f64,
    max_us: f64,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
}

impl Stats {
    fn compute(mut samples: Vec<f64>) -> Self {
        assert!(!samples.is_empty());
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let count = samples.len();
        let total_us: f64 = samples.iter().sum();
        let min_us = samples[0];
        let max_us = samples[count - 1];
        let mean_us = total_us / count as f64;
        let p50_us = samples[count * 50 / 100];
        let p95_us = samples[count * 95 / 100];
        let p99_us = samples[count * 99 / 100];

        Self {
            count,
            min_us,
            max_us,
            mean_us,
            p50_us,
            p95_us,
            p99_us,
        }
    }
}

/// `budget_us` is the audio duration of what one sample pushed, so every
/// ratio below is per sample pushed.
fn print_stats(title: &str, stats: &Stats, budget_us: f64) {
    println!("---------------------------------------------------------------");
    println!("  Scenario: {}", title);
    println!("---------------------------------------------------------------");
    println!("  Pushes sampled:     {}", stats.count);
    println!(
        "  Audio per push:     {:>8.2} µs ({:.2} ms)",
        budget_us,
        budget_us / 1000.0
    );
    println!(
        "  Mean:               {:>8.2} µs  = {:.3}x of the audio it covers",
        stats.mean_us,
        stats.mean_us / budget_us
    );
    println!(
        "  Min / P50:          {:>8.2} / {:.2} µs",
        stats.min_us, stats.p50_us
    );
    println!(
        "  P95 / P99:          {:>8.2} / {:.2} µs",
        stats.p95_us, stats.p99_us
    );
    println!(
        "  Max:                {:>8.2} µs = {:.3}x (one burst; encoder output comes once per 128 units)",
        stats.max_us,
        stats.max_us / budget_us
    );
    println!();
}

/// A deterministic 16-channel stand-in for a 7.1.4 programme with objects:
/// eight bed channels (tones with a slow envelope plus a little noise, a
/// low-passed LFE), four height channels of quieter tones plus noise, and four
/// object channels, two of them white noise at a high level, which is what
/// prediction can do least with. 24-bit, 48 kHz.
struct Synth {
    frames: Vec<Vec<Vec<i32>>>, // [frame][channel][sample]
}

impl Synth {
    #[allow(clippy::needless_range_loop)]
    fn new(frame_count: usize, samples: usize) -> Self {
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let mut noise = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            ((rng >> 40) as i64 - (1 << 23)) as f64 / (1 << 23) as f64
        };
        let mut t = 0u64;
        let mut lfe = 0.0f64;
        let mut frames = Vec::new();
        for _ in 0..frame_count {
            let mut ch = vec![vec![0i32; samples]; 16];
            for s in 0..samples {
                let x = (t + s as u64) as f64 / 48000.0;
                let env = 0.6 + 0.4 * (x * 1.3).sin();
                for c in 0..16 {
                    let v = match c {
                        3 => {
                            lfe += 0.02
                                * (noise() * 0.5 + (x * 2.0 * std::f64::consts::PI * 45.0).sin()
                                    - lfe);
                            lfe * 0.5
                        }
                        0..=7 => {
                            let f = 110.0 * (c as f64 + 1.0);
                            env * (0.25 * (x * 2.0 * std::f64::consts::PI * f).sin()
                                + 0.12 * (x * 2.0 * std::f64::consts::PI * f * 2.01).sin()
                                + 0.05 * noise())
                        }
                        8..=11 => {
                            0.08 * (x * 2.0 * std::f64::consts::PI * (500.0 + 90.0 * c as f64))
                                .sin()
                                + 0.03 * noise()
                        }
                        12 | 13 => {
                            0.15 * (x * 2.0 * std::f64::consts::PI * (300.0 + 40.0 * c as f64))
                                .sin()
                                * env
                        }
                        _ => 0.5 * noise(),
                    };
                    ch[c][s] = (v * 8_388_607.0 * 0.5) as i32;
                }
            }
            t += samples as u64;
            frames.push(ch);
        }
        Self { frames }
    }

    fn push(&self, ctx: &mut Context, i: usize, pts: f64) {
        let f = &self.frames[i % self.frames.len()];
        let bed: [&[i32]; 8] = std::array::from_fn(|c| f[c].as_slice());
        let ext: [&[i32]; 8] = std::array::from_fn(|c| f[8 + c].as_slice());
        ctx.push_pcm(&bed, &ext, pts).unwrap();
    }
}

fn run_synth(title: &str, effort: u32, synth: &Synth, samples_per_frame: usize, iterations: usize) {
    run_generic(
        title,
        effort,
        samples_per_frame,
        iterations,
        |ctx, i, pts| synth.push(ctx, i, pts),
    );
}

fn run_generic(
    title: &str,
    effort: u32,
    samples_per_frame: usize,
    iterations: usize,
    mut push: impl FnMut(&mut Context, usize, f64),
) {
    let frame_us = samples_per_frame as f64 * 1.0e6 / 48000.0;
    let mut ctx = Context::new(48000, effort).expect("context");
    ctx.set_key(&[0x42u8; 32]);
    let mut out_buf = vec![0u8; 65536];
    let mut pts = 0.0f64;
    let mut pts_in = 0.0f64;

    // Warm-up: run past the encoder's start-up so the pipeline is full.
    for i in 0..800 {
        push(&mut ctx, i, pts_in);
        pts_in += frame_us;
        while ctx.pull_truehd(&mut out_buf, &mut pts).unwrap() > 0 {}
    }

    let mut samples = Vec::with_capacity(iterations);
    let mut bytes = 0usize;
    let mut audio_us = 0.0f64;
    let run = Instant::now();
    // Paced at PACE times real time. The pipeline runs beside the caller, so
    // pushing flat out would only measure how fast the queues fill; paced, the
    // time of a push is the time the caller really spends on it, and a
    // pipeline that cannot keep up shows as pushes that wait.
    let period = std::time::Duration::from_secs_f64(frame_us / 1.0e6 / PACE);
    let mut due = Instant::now();
    for i in 0..iterations {
        due += period;
        while Instant::now() < due {
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
        let t0 = Instant::now();
        push(&mut ctx, 800 + i, pts_in);
        pts_in += frame_us;
        audio_us += frame_us;
        while let Ok(n) = ctx.pull_truehd(&mut out_buf, &mut pts) {
            if n == 0 {
                break;
            }
            bytes += n;
        }
        samples.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    // The pipeline runs beside the caller, so the work is only done once it
    // has been drained: the sustained figure includes the flush.
    let t_flush = Instant::now();
    ctx.flush();
    let drain_ms = t_flush.elapsed().as_secs_f64() * 1000.0;
    while let Ok(n) = ctx.pull_truehd(&mut out_buf, &mut pts) {
        if n == 0 {
            break;
        }
        bytes += n;
    }
    let sustained = run.elapsed().as_secs_f64() * 1.0e6;

    let pairs: Vec<f64> = samples
        .chunks(2)
        .filter(|p| p.len() == 2)
        .map(|p| p[0] + p[1])
        .collect();
    let stats = Stats::compute(samples);
    print_stats(title, &stats, frame_us);
    let w = Stats::compute(pairs);
    println!(
        "  Per {:.1} ms window: mean {:.1} / P99 {:.1} / max {:.1} µs",
        2.0 * frame_us / 1000.0,
        w.mean_us,
        w.p99_us,
        w.max_us
    );
    println!(
        "  Wall (paced {PACE}x): {:.1} ms for {:.1} ms of audio pushed; flush after the last push took {:.1} ms",
        sustained / 1000.0,
        audio_us / 1000.0,
        drain_ms
    );
    // Throughput: the same pushes flat out, to the end of the flush.
    let t_all = Instant::now();
    let flat = 1000usize;
    for i in 0..flat {
        push(&mut ctx, 1800 + i, pts_in);
        pts_in += frame_us;
        while ctx.pull_truehd(&mut out_buf, &mut pts).unwrap() > 0 {}
    }
    ctx.flush();
    while ctx.pull_truehd(&mut out_buf, &mut pts).unwrap() > 0 {}
    let flat_us = t_all.elapsed().as_secs_f64() * 1.0e6;
    println!(
        "  Throughput:         {:.3}x real time flat out ({:.1} ms for {:.1} ms of audio, flush included)",
        flat_us / (flat as f64 * frame_us),
        flat_us / 1000.0,
        flat as f64 * frame_us / 1000.0
    );
    println!("  TrueHD produced:    {} bytes\n", bytes);
}

/// How much faster than real time the timed pushes arrive.
const PACE: f64 = 4.0;

/// Effort levels the benchmark covers: (name, level).
const EFFORTS: &[(&str, u32)] = &[("Realtime", 0), ("Fast", 1), ("Full", 2)];

fn main() {
    println!("\n===============================================================");
    println!("   hz-stream pipeline timing");
    println!(
        "   Ran on: {} / {}, {} logical CPUs.",
        std::env::consts::OS,
        std::env::consts::ARCH,
        std::thread::available_parallelism().map_or(0, |n| n.get())
    );
    println!("===============================================================\n");

    let synth = Synth::new(200, 512);
    println!(
        "Synthetic 16-channel input: 7.1 bed + 4 heights + 2 tonal objects + 2 white-noise objects, 24-bit.\n"
    );
    for (name, effort) in EFFORTS {
        run_synth(
            &format!("S. 16-ch synthetic, {name} effort"),
            *effort,
            &synth,
            512,
            2000,
        );
    }
}
