use super::*;
use hz_dca::encoder::{Config as CoreConfig, Encoder as CoreEncoder, FRAME_SAMPLES, Layout};

/// Interleaved 5.1 tones, one frame's worth: a different pitch on each
/// channel, so nothing a stage mixes up goes unnoticed.
fn tone_frame(channels: usize, frame: usize) -> Vec<i32> {
    tone_frame_where(channels, frame, |_| true)
}

/// [`tone_frame`], with the channels `keep` refuses left silent.
fn tone_frame_where(channels: usize, frame: usize, keep: impl Fn(usize) -> bool) -> Vec<i32> {
    (0..FRAME_SAMPLES * channels)
        .map(|i| {
            let (n, c) = (i / channels, i % channels);
            if !keep(c) {
                return 0;
            }
            let t = (frame * FRAME_SAMPLES + n) as f64 / 48000.0;
            (((t * (220.0 + 110.0 * c as f64) * std::f64::consts::TAU).sin() * 0.25 * 8_388_607.0)
                as i32)
                & !0xff
        })
        .collect()
}

/// Ten core frames of a 5.1 tone at `rate`, each a frame of its own.
fn core_frames(rate: u32) -> Vec<Vec<u8>> {
    let mut encoder = CoreEncoder::new(CoreConfig {
        sample_rate: rate,
        ..CoreConfig::reference_five_one()
    })
    .expect("a core encoder");
    (0..10)
        .map(|frame| {
            let mut bytes = Vec::new();
            encoder.encode(
                &tone_frame(Layout::FivePointOne.channels(), frame),
                &mut bytes,
            );
            let info = parse_header(&bytes).expect("the frame has a header");
            assert_eq!(info.frame_size, bytes.len());
            bytes
        })
        .collect()
}

/// A DTS-HD stream of `frames` frames, the core and the substream that
/// carries the lossless 7.1.4 beside it, cut into its frames.
fn master_frames(frames: usize) -> Vec<Vec<u8>> {
    master_frames_where(frames, |_| true)
}

/// [`master_frames`], with the input channels `keep` refuses left silent.
fn master_frames_where(frames: usize, keep: impl Fn(usize) -> bool) -> Vec<Vec<u8>> {
    use hz_dca::lossless::Layout as Wide;
    use hz_dca::master::{Config, Encoder};
    let channels = Wide::SevenPointOneFour.channels();
    let mut encoder =
        Encoder::new(Config::reference(Wide::SevenPointOneFour, 24)).expect("a master encoder");
    let mut stream = Vec::new();
    for frame in 0..frames {
        encoder
            .encode(&tone_frame_where(channels, frame, &keep), &mut stream)
            .expect("encodes");
    }
    encoder.finish(&mut stream).expect("finishes");
    let mut cut = Vec::new();
    let mut at = 0;
    while at < stream.len() {
        let info = parse_header(&stream[at..]).expect("a core header");
        let core_end = at + info.frame_size;
        let end = match exss_substream_size(&stream[core_end..]) {
            Some(size) if stream[core_end..].starts_with(&SUBSTREAM_SYNC) => core_end + size,
            _ => core_end,
        };
        cut.push(stream[at..end].to_vec());
        at = end;
    }
    cut
}

/// Every unit currently queued, with its pts.
fn drain(ctx: &mut Context) -> Vec<(Vec<u8>, f64)> {
    let mut out = Vec::new();
    let mut buf = vec![0u8; 65536];
    loop {
        let mut pts = 0.0;
        let n = ctx
            .pull_truehd(&mut buf, &mut pts)
            .expect("buffer large enough");
        if n == 0 {
            return out;
        }
        out.push((buf[..n].to_vec(), pts));
    }
}

fn frame_us(i: usize, base: f64) -> f64 {
    base + (i * 512) as f64 * 1.0e6 / 48000.0
}

/// `n` frames cycled from the fixture, each pushed alone with its pts.
fn push_frames(ctx: &mut Context, frames: &[Vec<u8>], n: usize, base: f64) {
    for i in 0..n {
        ctx.push_dts(&frames[i % frames.len()], frame_us(i, base))
            .unwrap();
    }
}

#[test]
fn every_sample_comes_out_and_nothing_is_added() {
    let frames = core_frames(48000);
    let mut ctx = Context::new(48000, 0).unwrap();
    // 13 frames of 512 samples: 6656 = 166 units + 16 samples.
    push_frames(&mut ctx, &frames, 13, 0.0);
    assert!(ctx.flush() > 0);
    assert_eq!(ctx.stats().samples_fed, 13 * 512);
    assert_eq!(ctx.stats().silence_frames, 0);
    let units = drain(&mut ctx);
    // 166 whole units and the padded last one.
    assert_eq!(units.len(), 167);
    let stream: Vec<u8> = units.iter().flat_map(|(u, _)| u.clone()).collect();
    let parsed = hz_mlp::reader::stream(&stream).expect("whole TrueHD stream parses");
    assert_eq!(parsed.len(), 167);
    assert_eq!(ctx.stats().units_out, 167);
}

#[test]
fn frame_remainder_carries_over() {
    // 5 frames = 2560 samples = 64 units exactly; 3 frames = 38.4 units.
    let frames = core_frames(48000);
    let mut ctx = Context::new(48000, 0).unwrap();
    // A frame stays queued until the bytes after it show no substream
    // follows, so after n pushes n - 1 frames are in.
    push_frames(&mut ctx, &frames, 4, 0.0);
    assert!(ctx.sync());
    assert_eq!(ctx.stats().samples_fed, 1536);
    assert_eq!(ctx.stats().units_encoded, 38);
    assert_eq!(ctx.stats().samples_carried, 16);
    push_frames(&mut ctx, &frames, 2, 0.0);
    assert!(ctx.sync());
    assert_eq!(ctx.stats().samples_fed, 2560);
    assert_eq!(ctx.stats().units_encoded, 64);
    assert_eq!(ctx.stats().samples_carried, 0);
}

#[test]
fn packet_boundaries_do_not_change_the_output() {
    let frames = core_frames(48000);
    let all: Vec<u8> = (0..60).flat_map(|i| frames[i % 10].clone()).collect();

    let mut whole = Context::new(48000, 0).unwrap();
    whole.push_dts(&all, 0.0).unwrap();
    whole.flush();
    let a: Vec<u8> = drain(&mut whole).into_iter().flat_map(|(u, _)| u).collect();

    let mut chunked = Context::new(48000, 0).unwrap();
    for chunk in all.chunks(97) {
        chunked.push_dts(chunk, f64::NAN).unwrap();
    }
    chunked.flush();
    let b: Vec<u8> = drain(&mut chunked)
        .into_iter()
        .flat_map(|(u, _)| u)
        .collect();

    assert!(!a.is_empty());
    assert_eq!(a, b);
}

#[test]
fn pts_is_monotonic_matches_the_audio_and_never_stalls() {
    let frames = core_frames(48000);
    let base = 1_000_000.0;
    let mut ctx = Context::new(48000, 0).unwrap();
    let mut pulled = Vec::new();
    for i in 0..200 {
        ctx.push_dts(&frames[i % 10], frame_us(i, base)).unwrap();
        assert!(ctx.sync());
        pulled.extend(drain(&mut ctx));
    }
    // Output has started well before the end (the old bookkeeping
    // stopped yielding timestamps after ~15 frames).
    assert!(
        pulled.len() > 1000,
        "only {} units before flush",
        pulled.len()
    );
    ctx.flush();
    pulled.extend(drain(&mut ctx));
    assert_eq!(pulled.len(), (200 * 512usize).div_ceil(40));
    let mut last = f64::NEG_INFINITY;
    for (k, (_, pts)) in pulled.iter().enumerate() {
        assert!(pts.is_finite(), "unit {k} has no pts");
        assert!(*pts > last, "unit {k} not increasing");
        last = *pts;
        let expect = base + (k * 40) as f64 * 1.0e6 / 48000.0;
        assert!((pts - expect).abs() < 1.0, "unit {k}: {pts} vs {expect}");
    }
    assert_eq!(ctx.stats().pts_pending, 0);
}

#[test]
fn missing_or_invalid_pts_is_nan_not_minus_one() {
    let frames = core_frames(48000);
    let mut ctx = Context::new(48000, 0).unwrap();
    for i in 0..60 {
        let bad = [-1.0, f64::NAN, 1.8e19][i % 3];
        ctx.push_dts(&frames[i % 10], bad).unwrap();
    }
    ctx.flush();
    let units = drain(&mut ctx);
    assert!(!units.is_empty());
    assert!(units.iter().all(|(_, pts)| pts.is_nan()));
}

#[test]
fn parser_resynchronises_after_garbage() {
    let frames = core_frames(48000);
    let mut ctx = Context::new(48000, 0).unwrap();
    let mut garbage = vec![0x55u8; 300];
    // A false sync word with a header that does not parse, then more junk.
    garbage.extend_from_slice(&CORE_SYNC);
    garbage.extend_from_slice(&[0xff; 40]);
    garbage.extend_from_slice(&[0x00; 200]);
    for (i, frame) in frames.iter().enumerate().take(4) {
        ctx.push_dts(frame, frame_us(i, 0.0)).unwrap();
    }
    ctx.push_dts(&garbage, f64::NAN).unwrap();
    for (i, frame) in frames.iter().enumerate().skip(4) {
        ctx.push_dts(frame, frame_us(i, 0.0)).unwrap();
    }
    ctx.flush();
    let stats = ctx.stats();
    assert_eq!(stats.frames_decoded + stats.silence_frames, 10);
    assert_eq!(stats.samples_fed, 10 * 512);
    assert!(stats.resyncs >= 1);
    assert!(stats.bytes_skipped >= 300);
    assert!(
        stats.input_held < 64,
        "buffer kept {} bytes",
        stats.input_held
    );
}

#[test]
fn input_without_any_sync_does_not_accumulate() {
    let mut ctx = Context::new(48000, 0).unwrap();
    for _ in 0..200 {
        ctx.push_dts(&[0xa5u8; 10_000], f64::NAN).unwrap();
    }
    assert!(ctx.sync());
    assert!(ctx.stats().input_held <= 3);
    assert_eq!(ctx.stats().samples_fed, 0);
}

#[test]
fn a_damaged_frame_is_replaced_by_silence_of_its_length() {
    let mut frames = core_frames(48000);
    let len = frames[5].len();
    frames[5][16..len].fill(0);
    let mut ctx = Context::new(48000, 0).unwrap();
    push_frames(&mut ctx, &frames, 10, 0.0);
    ctx.flush();
    let stats = ctx.stats();
    assert_eq!(stats.frames_decoded + stats.silence_frames, 10);
    assert_eq!(stats.samples_fed, 10 * 512);
}

#[test]
fn a_stream_at_another_rate_is_refused_with_silence_of_its_length() {
    let frames = core_frames(44100);
    let n = parse_header(&frames[0]).unwrap().samples_per_channel();
    let mut ctx = Context::new(48000, 0).unwrap();
    for f in &frames {
        // Decoding is behind the push, so the refusal is reported by the
        // push after the one that carried the frame.
        let _ = ctx.push_dts(f, f64::NAN);
    }
    ctx.flush();
    assert_eq!(ctx.stats().silence_frames, 10);
    assert!(ctx.push_dts(&frames[0], f64::NAN).is_err());
    assert!(ctx.push_dts(&frames[0], f64::NAN).is_ok(), "reported once");
    assert_eq!(ctx.stats().samples_fed, (10 * n) as u64);
}

#[test]
fn reset_drops_audio_the_encoder_was_holding() {
    let frames = core_frames(48000);
    let mut ctx = Context::new(48000, 0).unwrap();
    push_frames(&mut ctx, &frames, 200, 0.0);
    assert!(ctx.sync());
    assert!(ctx.queued() > 0 || ctx.stats().units_out > 0);
    ctx.reset();
    assert_eq!(drain(&mut ctx).len(), 0);
    // After the seek only the new material comes out, 3 frames of it.
    let seek = 50_000_000.0;
    push_frames(&mut ctx, &frames, 3, seek);
    ctx.flush();
    let units = drain(&mut ctx);
    assert_eq!(units.len(), (3 * 512usize).div_ceil(40));
    assert!((units[0].1 - seek).abs() < 1.0);
    // A fresh stream starts with its major sync.
    assert!(units[0].0.windows(4).any(|w| w == [0xf8, 0x72, 0x6f, 0xba]));
}

#[test]
fn flush_then_more_input_starts_a_new_stream() {
    let frames = core_frames(48000);
    let mut ctx = Context::new(48000, 0).unwrap();
    push_frames(&mut ctx, &frames, 5, 0.0);
    ctx.flush();
    let first = drain(&mut ctx).len();
    push_frames(&mut ctx, &frames, 5, 0.0);
    ctx.flush();
    assert_eq!(first, drain(&mut ctx).len());
}

/// A substream header, up to its size field, padded with zeros to `size`.
fn fake_exss(size: usize) -> Vec<u8> {
    let mut bits = format!("000{:08b}{:016b}", 0x0f, size - 1);
    while bits.len() % 8 != 0 {
        bits.push('0');
    }
    let mut data = vec![0x64, 0x58, 0x20, 0x25, 0x00];
    for chunk in bits.as_bytes().chunks(8) {
        data.push(u8::from_str_radix(std::str::from_utf8(chunk).unwrap(), 2).unwrap());
    }
    data.resize(size, 0);
    data
}

fn stage() -> DecodeStage {
    let (tx, _rx) = sync_channel::<Work>(2);
    DecodeStage::new(48000, Arc::new(Shared::new()), tx)
}

#[test]
fn a_substream_cut_by_a_packet_boundary_is_waited_for() {
    let core = core_frames(48000).remove(0);
    let exss = fake_exss(300);
    let whole: Vec<u8> = core.iter().chain(exss.iter()).copied().collect();
    let core_end = core.len();
    for cut in [
        core_end + 4,
        core_end + 6,
        core_end + 8,
        core_end + 9,
        core_end + 150,
        whole.len() - 1,
    ] {
        let mut st = stage();
        st.input_buf = whole[..cut].to_vec();
        assert!(
            st.find_next_frame(false).is_none(),
            "cut at {cut} must wait"
        );
        assert_eq!(
            st.input_buf.len(),
            cut,
            "nothing of the frame is dropped while waiting"
        );
        // At the end of the stream the core alone is all there is.
        let found = st
            .find_next_frame(true)
            .expect("flush decodes what there is");
        assert!(found.exss.is_empty());
    }
    let mut st = stage();
    st.input_buf = whole.clone();
    let found = st.find_next_frame(false).expect("a whole frame");
    assert_eq!(found.core, 0..core_end);
    assert_eq!(found.exss, core_end..core_end + 300);
}

#[test]
fn dense_false_sync_words_cost_linear_time() {
    let mut st = stage();
    st.input_buf = CORE_SYNC.iter().copied().cycle().take(1 << 18).collect();
    let t = std::time::Instant::now();
    let _ = st.find_next_frame(false);
    assert!(
        t.elapsed() < Duration::from_millis(500),
        "took {:?}",
        t.elapsed()
    );
    assert!(
        st.input_buf.len() < 1024,
        "the junk is dropped, {} bytes left",
        st.input_buf.len()
    );
}

#[test]
fn a_flush_in_the_middle_of_a_unit_does_not_delay_the_next_stream() {
    let frames = core_frames(48000);
    let mut ctx = Context::new(48000, 0).unwrap();
    // Three frames are 1536 samples: 38 units and 16 samples over.
    push_frames(&mut ctx, &frames, 3, 0.0);
    ctx.flush();
    let first = drain(&mut ctx);
    assert_eq!(first.len(), 39);
    push_frames(&mut ctx, &frames, 3, 10_000_000.0);
    ctx.flush();
    let second = drain(&mut ctx);
    assert_eq!(second.len(), 39);
    let late = second[0].1 - 10_000_000.0;
    assert!(late.abs() < 1.0, "the second stream starts {late} us late");
    // And it stays on time.
    let last = second[38].1 - 10_000_000.0 - 38.0 * 40.0 * 1.0e6 / 48000.0;
    assert!(last.abs() < 1.0, "off by {last} us at the end");
}

#[test]
fn a_panic_in_the_decoder_costs_the_packet_not_the_stream() {
    let frames = core_frames(48000);
    let mut ctx = Context::new(48000, 0).unwrap();
    push_frames(&mut ctx, &frames, 3, 0.0);
    ctx.shared
        .inject_panic_decode
        .store(true, Ordering::Release);
    // The packet is taken, as a host that is told otherwise re-sends it.
    assert_eq!(ctx.push_dts(&frames[3], 0.0), Ok(frames[3].len()));
    assert!(ctx.sync());
    assert_eq!(ctx.stats().panics, 1);
    let before = ctx.stats().units_encoded;
    push_frames(&mut ctx, &frames, 4, 1_000_000.0);
    assert!(ctx.sync());
    assert!(ctx.stats().units_encoded > before, "decoding goes on");
    ctx.flush();
    assert!(!drain(&mut ctx).is_empty());
}

#[test]
fn a_panic_in_the_encoder_is_recovered_from_without_a_reset() {
    let frames = core_frames(48000);
    let mut ctx = Context::new(48000, 0).unwrap();
    push_frames(&mut ctx, &frames, 3, 0.0);
    assert!(ctx.sync());
    ctx.shared
        .inject_panic_encode
        .store(true, Ordering::Release);
    push_frames(&mut ctx, &frames, 1, 0.0);
    assert!(ctx.sync());
    assert_eq!(ctx.stats().panics, 1);
    push_frames(&mut ctx, &frames, 10, 2_000_000.0);
    ctx.flush();
    let units = drain(&mut ctx);
    assert!(
        units.len() >= 64,
        "the encoder writes again ({} units)",
        units.len()
    );
    // The audio the encoder lost is not in the timeline. The restart is
    // at the fourth frame (stream position 1536, pts 0), and the fifth,
    // at 2048 and 2 s, starts 8 samples into a unit.
    assert!(
        units[0].1.abs() < 1.0,
        "the first unit after the restart is at {}",
        units[0].1
    );
    let straddling = units[13].1;
    let expected = 2_000_000.0 + 8.0 * 1.0e6 / 48000.0;
    assert!(
        (straddling - expected).abs() < 1.0,
        "unit 13 is at {straddling}, not {expected}"
    );
    for w in units.windows(2).skip(14) {
        assert!(
            (w[1].1 - w[0].1 - 40.0e6 / 48000.0).abs() < 1.0,
            "units stay 40 samples apart"
        );
    }
}

#[test]
fn pull_into_a_small_buffer_asks_for_more_and_loses_nothing() {
    let frames = core_frames(48000);
    let mut ctx = Context::new(48000, 0).unwrap();
    push_frames(&mut ctx, &frames, 4, 0.0);
    ctx.flush();
    let mut tiny = [0u8; 8];
    let mut pts = 0.0;
    let needed = ctx.pull_truehd(&mut tiny, &mut pts).unwrap_err();
    assert!(needed > 8);
    assert_eq!(drain(&mut ctx).len(), (4 * 512usize).div_ceil(40));
}

/// A sixteen-channel noise frame, which is the output-heavy case.
fn noise_frames(count: usize) -> Vec<Vec<Vec<i32>>> {
    let mut rng = 0x2545_f491_4f6c_dd1du64;
    (0..count)
        .map(|_| {
            (0..16)
                .map(|_| {
                    (0..512)
                        .map(|_| {
                            rng ^= rng << 13;
                            rng ^= rng >> 7;
                            rng ^= rng << 17;
                            ((rng >> 42) as i32) - (1 << 21)
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

fn push_noise(ctx: &mut Context, frames: &[Vec<Vec<i32>>], base: f64) {
    for (i, f) in frames.iter().enumerate() {
        let bed: [&[i32]; 8] = std::array::from_fn(|c| f[c].as_slice());
        let ext: [&[i32]; 8] = std::array::from_fn(|c| f[8 + c].as_slice());
        ctx.push_pcm(&bed, &ext, frame_us(i, base)).unwrap();
    }
}

/// `count` frames of sixteen channels of full-scale 24-bit white noise,
/// which no lossless coder can fit in the declared rate.
fn noise24_frames(count: usize) -> Vec<Vec<Vec<i32>>> {
    let mut rng = 0x9e37_79b9_7f4a_7c15u64;
    (0..count)
        .map(|_| {
            (0..16)
                .map(|_| {
                    (0..512)
                        .map(|_| {
                            rng ^= rng << 13;
                            rng ^= rng >> 7;
                            rng ^= rng << 17;
                            ((rng >> 40) as i32) - (1 << 23)
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

/// Pull until the encoder has gone quiet, for content whose output would
/// fill the queue if left unpulled (a `sync` then never returns).
fn settle(ctx: &mut Context) {
    let (mut last, mut still) = (u64::MAX, 0);
    for _ in 0..400 {
        drain(ctx);
        std::thread::sleep(Duration::from_millis(40));
        let now = ctx.stats().units_encoded;
        still = if now == last { still + 1 } else { 0 };
        if still >= 5 {
            break;
        }
        last = now;
    }
    drain(ctx);
}

/// Like `push_noise`, pulling as it goes so the output queue never fills.
fn push_noise_pulling(ctx: &mut Context, frames: &[Vec<Vec<i32>>], base: f64) {
    for (i, chunk) in frames.chunks(5).enumerate() {
        push_noise(ctx, chunk, base + (i * 5 * 512) as f64 * 1.0e6 / 48000.0);
        drain(ctx);
    }
}

#[test]
fn content_over_the_declared_rate_is_counted_and_brought_back_under_it() {
    let frames = noise24_frames(400);
    let mut ctx = Context::new(48000, 0).unwrap();
    push_noise_pulling(&mut ctx, &frames[..200], 0.0);
    settle(&mut ctx);
    let early = ctx.stats();
    assert!(
        early.starved_units > 0,
        "24-bit noise on sixteen channels cannot fit the rate"
    );
    assert!(early.dropped_bits > 0, "the input is made to fit");
    drain(&mut ctx);
    push_noise_pulling(&mut ctx, &frames[200..], 1.0e7);
    settle(&mut ctx);
    let late = ctx.stats();
    drain(&mut ctx);
    eprintln!(
        "starved {} -> {}, dropped bits {} -> {}",
        early.starved_units, late.starved_units, early.dropped_bits, late.dropped_bits
    );
    assert!(late.dropped_bits <= MAX_DROPPED_BITS);
    // Once the clearing has taken hold, no unit is starved any more.
    push_noise_pulling(&mut ctx, &frames[..100], 2.0e7);
    settle(&mut ctx);
    assert_eq!(
        ctx.stats().starved_units,
        late.starved_units,
        "still starving after {} bits dropped",
        late.dropped_bits
    );
}

#[test]
fn content_that_fits_is_left_alone() {
    let frames = noise_frames(100);
    let mut ctx = Context::new(48000, 0).unwrap();
    push_noise_pulling(&mut ctx, &frames, 0.0);
    settle(&mut ctx);
    let stats = ctx.stats();
    assert_eq!((stats.starved_units, stats.dropped_bits), (0, 0));
}

/// The caller stops pulling while the encoder is far enough ahead to hit
/// the output limit: a flush still finishes, and loses nothing.
#[test]
fn a_flush_finishes_while_the_output_queue_is_full() {
    let frames = noise_frames(60);
    let mut ctx = Context::new(48000, 0).unwrap();
    push_noise(&mut ctx, &frames, 0.0);
    assert!(ctx.flush() > 0);
    let units = drain(&mut ctx);
    assert_eq!(units.len(), (60 * 512usize).div_ceil(40));
    let stream: Vec<u8> = units.iter().flat_map(|(u, _)| u.clone()).collect();
    assert_eq!(
        hz_mlp::reader::stream(&stream).expect("parses").len(),
        units.len()
    );
}

/// A reset with the pipeline backed up and stalled returns, and nothing
/// of the old stream comes out after it.
#[test]
fn a_reset_while_the_pipeline_is_backed_up_drops_it_all() {
    let frames = noise_frames(60);
    let mut ctx = Context::new(48000, 0).unwrap();
    push_noise(&mut ctx, &frames, 0.0);
    ctx.reset();
    assert_eq!(ctx.queued(), 0);
    assert_eq!(drain(&mut ctx).len(), 0);
    let seek = 9_000_000.0;
    push_noise(&mut ctx, &frames[..3], seek);
    ctx.flush();
    let units = drain(&mut ctx);
    assert_eq!(units.len(), (3 * 512usize).div_ceil(40));
    assert!((units[0].1 - seek).abs() < 1.0);
}

#[test]
fn hex_keys_are_parsed_strictly() {
    assert_eq!(parse_hex_key(" 00ff7A \n").unwrap(), vec![0x00, 0xff, 0x7a]);
    for bad in [
        "",
        "   ",
        "abc",
        "0x12ab",
        "12 34",
        "zz",
        "12ab\u{e9}",
        &"ab".repeat(MAX_KEY_BYTES + 1),
    ] {
        assert!(parse_hex_key(bad).is_err(), "{bad:?} must be rejected");
    }
    let mut ctx = Context::new(48000, 0).unwrap();
    assert!(ctx.set_key_hex("not a key").is_err());
    assert!(ctx.key.is_none());
    ctx.set_key_hex(&"5a".repeat(32)).unwrap();
    assert_eq!(ctx.key.as_deref(), Some(&[0x5au8; 32][..]));
    assert!(ctx.set_key_hex("5a5").is_err());
    assert_eq!(ctx.key.as_deref(), Some(&[0x5au8; 32][..]));
}

#[test]
fn the_key_survives_a_reset() {
    let frames = core_frames(48000);
    let signed = |reset: bool| {
        let mut ctx = Context::new(48000, 0).unwrap();
        ctx.set_key(&[7u8; 32]);
        if reset {
            push_frames(&mut ctx, &frames, 6, 0.0);
            ctx.reset();
        }
        push_frames(&mut ctx, &frames, 10, 0.0);
        ctx.flush();
        drain(&mut ctx)
            .into_iter()
            .flat_map(|(u, _)| u)
            .collect::<Vec<u8>>()
    };
    assert_eq!(signed(false), signed(true));
}

/// The stream a context writes for `frames`, pushed whole, as one run of
/// bytes, with the key if there is one.
fn transcode(frames: &[Vec<u8>], key: Option<&[u8]>) -> (Vec<u8>, Stats) {
    let mut ctx = Context::new(48000, 0).unwrap();
    if let Some(key) = key {
        ctx.set_key(key);
    }
    push_frames(&mut ctx, frames, frames.len(), 0.0);
    ctx.flush();
    let stats = ctx.stats();
    (
        drain(&mut ctx).into_iter().flat_map(|(u, _)| u).collect(),
        stats,
    )
}

#[test]
fn a_lossless_stream_is_carried_from_its_lossless_layer() {
    let frames = master_frames(12);
    let (stream, stats) = transcode(&frames, None);
    assert_eq!((stats.lossy_frames, stats.silence_frames), (0, 0));
    assert_eq!(stats.samples_fed, 12 * 512);
    assert_eq!(
        hz_mlp::reader::stream(&stream).expect("parses").len(),
        (12 * 512usize).div_ceil(40)
    );
}

/// The reference decoder of `hz-mlp`'s own tests, read back through here.
#[allow(dead_code)]
#[path = "../../hz-mlp/tests/decoder/mod.rs"]
mod decoder;

/// The payload of every unit that has one: under the object audio metadata
/// identifier, one to a unit, read back. The encoder writes the Evolution
/// frame; a payload that came in already wrapped in one would not read.
fn payloads(stream: &[u8]) -> Vec<hz_meta::oamd::ObjectAudioMetadata> {
    let units = hz_mlp::reader::stream(stream).expect("parses");
    units
        .iter()
        .filter_map(|u| u.evolution.as_ref())
        .filter(|e| !e.payloads.is_empty())
        .map(|extra| {
            assert_eq!(extra.payloads.len(), 1);
            assert_eq!(extra.payloads[0].id, OAMD);
            hz_meta::oamd::ObjectAudioMetadata::read(&extra.payloads[0].bytes).expect("reads back")
        })
        .collect()
}

/// What the metadata says of an element: its position, with `None` for the
/// low frequency channel and for an element that is not there.
fn placed(
    metadata: &hz_meta::oamd::ObjectAudioMetadata,
    element: usize,
) -> Option<hz_meta::oamd::Position> {
    let object = &metadata.blocks[0].objects[element];
    assert_eq!(object.active, element < 12, "element {element}");
    object.render.map(|render| render.position)
}

/// The 7.1.4 the DTS encoder is given, channel by channel in WAV order: where
/// each channel belongs in the room (`None` for the LFE).
const INPUT_AT: [Option<[f64; 3]>; 12] = [
    Some([-1.0, 1.0, 0.0]),  // L
    Some([1.0, 1.0, 0.0]),   // R
    Some([0.0, 1.0, 0.0]),   // C
    None,                    // LFE
    Some([-1.0, -1.0, 0.0]), // Lsr, the rear left
    Some([1.0, -1.0, 0.0]),  // Rsr
    Some([-1.0, 0.0, 0.0]),  // Lss, the side left
    Some([1.0, 0.0, 0.0]),   // Rss
    Some([-1.0, 1.0, 1.0]),  // TFL
    Some([1.0, 1.0, 1.0]),   // TFR
    Some([-1.0, -1.0, 1.0]), // TBL
    Some([1.0, -1.0, 1.0]),  // TBR
];

/// The input channel each of the stream's sixteen elements carries: the LFE
/// first, then L R C, the sides, the rears and the four heights; four spare.
const ELEMENT_INPUT: [Option<usize>; 16] = [
    Some(3),
    Some(0),
    Some(1),
    Some(2),
    Some(6),
    Some(7),
    Some(4),
    Some(5),
    Some(8),
    Some(9),
    Some(10),
    Some(11),
    None,
    None,
    None,
    None,
];

/// Transcode seven-point-one-four tones on the channels `keep` leaves alone,
/// decode the sixteen-element presentation and return what came out as
/// `out[element][sample]`, with the metadata the stream carried.
fn round_trip(
    keep: impl Fn(usize) -> bool + Copy,
) -> (Vec<Vec<i32>>, Vec<hz_meta::oamd::ObjectAudioMetadata>) {
    let frames = master_frames_where(12, keep);
    let (stream, _) = transcode(&frames, None);
    let widest = decoder::substreams(&stream).expect("a major sync") - 1;
    let decoded = decoder::decode(&stream, widest).expect("decodes");
    assert_eq!(decoded.channels, 16);
    let samples = 12 * FRAME_SAMPLES;
    assert!(decoded.frames() >= samples);
    let elements = (0..16)
        .map(|e| {
            (0..samples)
                .map(|t| decoded.samples[t * 16 + e])
                .collect::<Vec<i32>>()
        })
        .collect();
    (elements, payloads(&stream))
}

/// What the encoder was given for input channel `c`, as one run, as the
/// 24-bit samples the lossless layer carries (the DTS encoder takes them
/// left-justified in 32 bits).
fn input_tone(c: usize, keep: impl Fn(usize) -> bool) -> Vec<i32> {
    (0..12)
        .flat_map(|frame| {
            tone_frame_where(12, frame, &keep)
                .into_iter()
                .skip(c)
                .step_by(12)
                .map(|sample| sample >> 8)
                .collect::<Vec<i32>>()
        })
        .collect()
}

/// What a decoder finds after the last substream of every unit that has
/// anything there: one payload that describes the sixteen elements, and
/// element `i` of it describes plane `i`. A 7.1.4 stream with the bed
/// playing: the sixteen elements come back bit for bit as the planes it was
/// given, the LFE first, and the metadata puts each where its channel is.
#[test]
fn the_metadata_describes_the_planes_the_bed_comes_back_in() {
    let keep = |c: usize| c < 8;
    let (elements, metadata) = round_trip(keep);
    assert!(
        metadata.len() >= 12,
        "only {} units carry any",
        metadata.len()
    );
    for m in &metadata {
        assert!(m.lfe);
        assert_eq!(m.objects(), 16);
    }
    for (element, source) in ELEMENT_INPUT.iter().enumerate() {
        let expected =
            source.map_or_else(|| vec![0; elements[element].len()], |c| input_tone(c, keep));
        assert_eq!(elements[element], expected, "element {element}'s audio");
        assert!(element >= 8 || expected.iter().any(|v| *v != 0));
    }
    for m in &metadata {
        for (element, source) in ELEMENT_INPUT.iter().enumerate() {
            let want = source
                .and_then(|c| INPUT_AT[c])
                .map(hz_meta::oamd::Position::from_master);
            assert_eq!(placed(m, element), want, "element {element}'s place");
        }
    }
}

/// The same with the heights playing and the bed silent: the four extension
/// waveforms are elements 8 to 11, at the heights' places.
#[test]
fn the_heights_come_back_as_the_elements_the_metadata_places_them_at() {
    let keep = |c: usize| c >= 8;
    let (elements, metadata) = round_trip(keep);
    assert!(!metadata.is_empty());
    for (element, audio) in elements.iter().enumerate().skip(8).take(4) {
        let c = ELEMENT_INPUT[element].unwrap();
        assert_eq!(*audio, input_tone(c, keep), "height {element}");
        assert!(audio.iter().any(|v| *v != 0));
    }
    for (element, audio) in elements.iter().enumerate().skip(12) {
        assert!(audio.iter().all(|v| *v == 0), "{element}");
    }
    for m in &metadata {
        for element in 8..12 {
            let at = INPUT_AT[ELEMENT_INPUT[element].unwrap()].unwrap();
            assert_eq!(
                placed(m, element),
                Some(hz_meta::oamd::Position::from_master(at)),
                "element {element}"
            );
        }
    }
}

/// With the folds on, the 7.1 presentation names its channels L R C LFE Ls Rs
/// Lrs Rrs and gets them, though the stream's elements lead with the LFE.
#[test]
fn the_folded_seven_one_names_the_bed_in_its_own_order() {
    let keep = |c: usize| c < 8;
    let frames = master_frames_where(12, keep);
    let mut ctx = Context::with_options(
        48000,
        Options {
            folds: true,
            ..Options::default()
        },
    )
    .unwrap();
    push_frames(&mut ctx, &frames, frames.len(), 0.0);
    ctx.flush();
    let stream: Vec<u8> = drain(&mut ctx).into_iter().flat_map(|(u, _)| u).collect();
    assert_eq!(decoder::substreams(&stream).unwrap(), 4);
    let decoded = decoder::decode(&stream, 2).expect("decodes");
    assert_eq!(decoded.channels, 8);
    // L R C LFE Ls Rs Lrs Rrs, from the WAV-ordered input.
    for (channel, input) in [0usize, 1, 2, 3, 6, 7, 4, 5].into_iter().enumerate() {
        let wanted = input_tone(input, keep);
        let got: Vec<i32> = (0..wanted.len())
            .map(|t| decoded.samples[t * 8 + channel])
            .collect();
        assert_eq!(got, wanted, "channel {channel}");
    }
}

/// With the folds off (the default) the narrow presentations are stated all
/// the same, as selections of the bed: 2.0 is L R, 5.1 is L R C LFE Ls Rs and
/// 7.1 is the bed, each channel the very samples of its element, and the
/// sixteen elements are the planes bit for bit. Without a statement the
/// narrow ones would be the leading elements, LFE and L.
#[test]
fn the_narrow_presentations_select_the_named_channels_with_the_folds_off() {
    let keep = |c: usize| c < 8;
    let frames = master_frames_where(12, keep);
    let (stream, _) = transcode(&frames, None);
    assert_eq!(decoder::substreams(&stream).unwrap(), 4);
    // The input channel behind each channel of each presentation.
    let named: [&[usize]; 3] = [&[0, 1], &[0, 1, 2, 3, 6, 7], &[0, 1, 2, 3, 6, 7, 4, 5]];
    for (presentation, inputs) in named.into_iter().enumerate() {
        let decoded = decoder::decode(&stream, presentation).expect("decodes");
        assert_eq!(
            decoded.channels,
            inputs.len(),
            "presentation {presentation}"
        );
        for (channel, input) in inputs.iter().enumerate() {
            let wanted = input_tone(*input, keep);
            assert!(wanted.iter().any(|v| *v != 0));
            let got: Vec<i32> = (0..wanted.len())
                .map(|t| decoded.samples[t * inputs.len() + channel])
                .collect();
            assert_eq!(got, wanted, "presentation {presentation} channel {channel}");
        }
    }
    // The widest is the elements themselves.
    let (elements, _) = round_trip(keep);
    for (element, source) in ELEMENT_INPUT.iter().enumerate().take(8) {
        assert_eq!(elements[element], input_tone(source.unwrap(), keep));
    }
}

/// `count` frames of sixteen channels that share one slow signal, each at
/// its own gain, with a little independent noise on top: correlated the way
/// a real bed is, which the single tone per plane above never is.
fn correlated_frames(count: usize, level: f64) -> Vec<Vec<Vec<i32>>> {
    let mut rng = 0x1d87_2b41_a5c3_e9f7u64;
    let gain = |c: usize| 0.55 + 0.03 * c as f64;
    (0..count)
        .map(|frame| {
            (0..16)
                .map(|c| {
                    (0..512)
                        .map(|t| {
                            rng ^= rng << 13;
                            rng ^= rng >> 7;
                            rng ^= rng << 17;
                            let noise = ((rng >> 40) as f64 / (1u64 << 24) as f64) - 0.5;
                            let n = (frame * 512 + t) as f64;
                            let common = (n * 0.004).sin() + 0.5 * (n * 0.0011).sin();
                            let x = level * (gain(c) * common + 0.15 * noise);
                            (x * (1 << 22) as f64).round() as i32
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

/// The narrow presentations are exact on correlated content, not only on the
/// tones above: the encoder's decorrelation of the early channels would
/// rewrite the rows a 2.0, 5.1 or 7.1 decoder hands back, so it is off.
/// Every channel of every presentation is the very plane of the element it
/// names, and so are the sixteen elements, with nothing starved.
#[test]
fn the_narrow_presentations_are_exact_on_correlated_content() {
    // Six restart intervals of 5120 samples, a loud passage.
    let frames = correlated_frames(60, 0.9);
    let mut ctx = Context::new(48000, 0).unwrap();
    let mut stream = Vec::new();
    for (i, chunk) in frames.chunks(5).enumerate() {
        push_noise(&mut ctx, chunk, (i * 5 * 512) as f64 * 1.0e6 / 48000.0);
        stream.extend(drain(&mut ctx).into_iter().flat_map(|(u, _)| u));
    }
    ctx.flush();
    let stats = ctx.stats();
    stream.extend(drain(&mut ctx).into_iter().flat_map(|(u, _)| u));
    assert_eq!((stats.starved_units, stats.dropped_bits), (0, 0));
    assert_eq!(decoder::substreams(&stream).unwrap(), 4);
    let plane =
        |element: usize| -> Vec<i32> { frames.iter().flat_map(|f| f[element].clone()).collect() };
    // The element behind each channel of each presentation, in the order
    // L R, L R C LFE Ls Rs, L R C LFE Ls Rs Lrs Rrs, and then all sixteen.
    let sixteen: Vec<usize> = (0..16).collect();
    let named: [&[usize]; 4] = [
        &[L, R],
        &[L, R, C, LFE, LS, RS],
        &[L, R, C, LFE, LS, RS, LRS, RRS],
        &sixteen,
    ];
    for (presentation, elements) in named.into_iter().enumerate() {
        let decoded = decoder::decode(&stream, presentation).expect("decodes");
        assert_eq!(
            decoded.channels,
            elements.len(),
            "presentation {presentation}"
        );
        for (channel, element) in elements.iter().enumerate() {
            let wanted = plane(*element);
            assert!(wanted.iter().any(|v| *v != 0));
            assert!(decoded.frames() >= wanted.len());
            let wrong = (0..wanted.len())
                .filter(|&t| decoded.samples[t * elements.len() + channel] != wanted[t])
                .count();
            assert_eq!(
                wrong, 0,
                "presentation {presentation} channel {channel}: {wrong} samples differ"
            );
        }
    }
}

/// A DTS frame is 512 samples and a unit 40, so a frame starts part of the
/// way into a unit most of the time. The metadata that describes a frame
/// takes effect at the sample the frame starts at, not at the start of the
/// unit it rides in: for every payload, the unit's start plus its offset is
/// the first sample of a frame, to the sample.
#[test]
fn the_metadata_takes_effect_where_its_frame_starts() {
    let frames = master_frames_where(12, |c| c < 8);
    let (stream, _) = transcode(&frames, None);
    let units = hz_mlp::reader::stream(&stream).expect("parses");
    let mut starts = Vec::new();
    for (unit, parsed) in units.iter().enumerate() {
        let Some(extra) = parsed.evolution.as_ref() else {
            continue;
        };
        for payload in &extra.payloads {
            let at = unit * 40 + payload.sample_offset.unwrap_or(0) as usize;
            assert_eq!(at % FRAME_SAMPLES, 0, "unit {unit}: sample {at}");
            starts.push(at / FRAME_SAMPLES);
        }
    }
    // Every frame, once, in order.
    assert_eq!(starts, (0..12).collect::<Vec<_>>());
}

/// The heights playing, with the stated fold of each in the bed: the bed
/// comes back as the DTS encoder was given it, bit for bit, and the heights
/// are the elements of their own. Before the bed was cleaned they were in
/// both.
#[test]
fn the_bed_comes_back_without_the_heights_folded_into_it() {
    let keep = |_: usize| true;
    let (elements, _) = round_trip(keep);
    for (element, source) in ELEMENT_INPUT.iter().enumerate().take(12) {
        let wanted = input_tone(source.unwrap(), keep);
        assert!(wanted.iter().any(|v| *v != 0));
        assert_eq!(elements[element], wanted, "element {element}");
    }
}

/// The narrow presentations come from the cleaned bed: with the heights
/// playing, 7.1 is the bed as it was before the heights were folded in.
#[test]
fn the_narrow_presentations_are_cut_from_the_cleaned_bed() {
    let keep = |_: usize| true;
    let frames = master_frames_where(12, keep);
    let (stream, _) = transcode(&frames, None);
    let decoded = decoder::decode(&stream, 2).expect("decodes");
    assert_eq!(decoded.channels, 8);
    for (channel, input) in [0usize, 1, 2, 3, 6, 7, 4, 5].into_iter().enumerate() {
        let wanted = input_tone(input, keep);
        let got: Vec<i32> = (0..wanted.len())
            .map(|t| decoded.samples[t * 8 + channel])
            .collect();
        assert_eq!(got, wanted, "channel {channel}");
    }
}

#[test]
fn the_key_signs_the_frames_it_was_given_for_and_a_wrong_one_does_not() {
    let frames = master_frames(12);
    let key = [0x42u8; 32];
    let (stream, _) = transcode(&frames, Some(&key));
    let units = hz_mlp::reader::stream(&stream).expect("parses");
    let signed = hz_core::hmac::HmacSha256::new(&key);
    let other = hz_core::hmac::HmacSha256::new(&[0x43u8; 32]);
    let (mut at, mut checked) = (0, 0);
    for unit in &units {
        if unit
            .evolution
            .as_ref()
            .is_some_and(|e| !e.payloads.is_empty())
        {
            assert_eq!(unit.protection_verifies(&signed, &stream[at..]), Some(true));
            assert_eq!(unit.protection_verifies(&other, &stream[at..]), Some(false));
            checked += 1;
        }
        at += unit.bytes;
    }
    assert!(checked >= 12);
}

#[test]
fn folds_are_off_unless_asked_for() {
    let frames = noise_frames(8);
    let stream = |options: Options| {
        let mut ctx = Context::with_options(48000, options).unwrap();
        push_noise(&mut ctx, &frames, 0.0);
        ctx.flush();
        let stream: Vec<u8> = drain(&mut ctx).into_iter().flat_map(|(u, _)| u).collect();
        assert!(hz_mlp::reader::stream(&stream).is_ok());
        stream
    };
    assert!(!Options::default().folds);
    let plain = stream(Options::default());
    assert_eq!(
        plain,
        stream(Options {
            folds: false,
            ..Options::default()
        })
    );
    // The narrow presentations are folds of the bed, not selections of it.
    assert_ne!(
        plain,
        stream(Options {
            folds: true,
            ..Options::default()
        })
    );
}

/// Frames and packets go round a loop of a few buffers; they are not
/// allocated afresh for each one.
#[test]
fn frames_and_packets_are_recycled() {
    // Decoded frames, on their way from the decode thread to the encode
    // thread: bounded by the queue between them.
    let frames = master_frames(60);
    let mut ctx = Context::new(48000, 0).unwrap();
    for (i, chunk) in frames.chunks(4).enumerate() {
        push_frames(&mut ctx, chunk, chunk.len(), i as f64 * 1.0e5);
        assert!(ctx.sync());
        drain(&mut ctx);
    }
    assert_eq!(ctx.stats().frames_decoded, 60);
    let fresh = ctx.shared.planes.fresh_count();
    assert!(
        fresh <= 2 * SPARE_FRAMES,
        "{fresh} plane buffers for 60 frames"
    );
    assert!(ctx.shared.planes.idle_count() >= 1, "the buffers come back");
    // Packets are bounded by the input queue, which is far more than a
    // caller that waits now and then keeps in flight.
    let fresh = ctx.shared.packets.fresh_count();
    assert!(fresh < 60, "{fresh} packet buffers for 60 packets");
    assert!(ctx.shared.packets.idle_count() >= 1);

    // Frames pushed as decoded audio wait in the input queue instead.
    let noise = noise_frames(300);
    let mut ctx = Context::new(48000, 0).unwrap();
    push_noise_pulling(&mut ctx, &noise, 0.0);
    settle(&mut ctx);
    let fresh = ctx.shared.planes.fresh_count();
    assert!(
        fresh <= INPUT_QUEUE + SPARE_FRAMES,
        "{fresh} plane buffers for 300 frames"
    );
}
