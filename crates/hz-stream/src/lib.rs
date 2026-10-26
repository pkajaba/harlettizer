//! Real-time DTS to TrueHD conversion for a player.
//!
//! Pushes DTS / DTS-HD / DTS:X elementary-stream packets in, pulls TrueHD
//! access units out, one per call. Nothing touches the disk.
//!
//! Timestamps are in microseconds. A pushed timestamp that is negative, not
//! finite, or implausibly large counts as "none". A pulled unit whose time is
//! not known gets NaN, never a number that looks valid.
//!
//! The TrueHD stream is a lossless carrier of whatever the decoder produced:
//! the timestamp of an output unit is the time of the audio it carries, so
//! the encoder's own look-ahead (a few hundred milliseconds before the first
//! bytes appear) delays when data arrives, not what its timestamps say.
//!
//! What the stream carries is the sixteen elements and one block of object
//! audio metadata per frame that describes them in the order they are given:
//! the low frequency channel, the seven bed channels at their speakers, then
//! the extension waveforms of a DTS:X frame (either profile: the standard
//! four heights, or an alternate profile's objects and heights) at the
//! positions the frame states. The bed is the lossless bed with the carried
//! waveforms taken out again (`bed.rs`), so nothing is played twice; a
//! waveform whose fold cannot be taken out stays in the bed and is not
//! carried, and so does the one waveform of a ninth that the eight elements
//! have no room for. A frame with no extension describes the bed alone. The
//! 2.0, 5.1 and 7.1 presentations select channels of the cleaned bed. The
//! encoder signs the metadata when it was given a key; nothing else is
//! signed.

use std::collections::VecDeque;
use std::ops::Range;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

mod bed;
pub mod cores;
mod oamd;
mod pool;
#[cfg(test)]
mod tests;

use dca::parser::ParseError;
use dca::{HdDecoder, HdError, HdFrame, exss_substream_size, parse_header};
use hz_meta::emdf::OAMD;
use hz_mlp::format::{Config as MlpConfig, SampleBits};
use hz_mlp::hierarchy::Presentation;
use hz_mlp::{Effort, Encoder as MlpEncoder};
use pool::Pool;

const CORE_SYNC: [u8; 4] = [0x7f, 0xfe, 0x80, 0x01];
const SUBSTREAM_SYNC: [u8; 4] = [0x64, 0x58, 0x20, 0x25];
/// Effort levels of [`Context::new`] and `hz_stream_create`. Zero is the
/// default and the one a live player wants: the cheapest encoder that still
/// writes a conformant stream. Any level above [`EFFORT_FAST`] is full effort.
pub const EFFORT_REALTIME: u32 = 0;
pub const EFFORT_FAST: u32 = 1;
pub const EFFORT_FULL: u32 = 2;
/// Samples per TrueHD access unit at 48 kHz.
pub const BASE_FRAME_SIZE: usize = 40;
const CHANNELS: usize = 16;
/// Input that cannot make progress is dropped past this size.
const MAX_INPUT_BUFFER: usize = 1 << 20;
/// A substream larger than this is not a real one (the field allows 1 MiB);
/// the core alone plays instead of waiting for it.
const MAX_EXSS_SIZE: usize = 1 << 17;
/// A core frame is never smaller than its header nor larger than this.
const MIN_CORE_FRAME: usize = 95;
const MAX_CORE_FRAME: usize = 16384;
/// Buffers kept for reuse (see [`pool`]): the frames in flight between the
/// stages, one being written and one being read, with room to spare.
const SPARE_FRAMES: usize = FRAME_QUEUE + 4;
/// Packets in flight are bounded by the input queue.
const SPARE_PACKETS: usize = INPUT_QUEUE + 4;
/// The most low bits that may be cleared to keep to the declared rate.
const MAX_DROPPED_BITS: u32 = 4;
/// Samples the encoder takes to report on what it was fed: four restart
/// intervals of 5120, and one more for margin.
const REPORT_LAG_SAMPLES: u64 = 5 * 5120;
/// How many unconsumed timestamps are kept.
const MAX_PTS_QUEUE: usize = 1 << 16;
/// Timestamps from this value up are a host's "no pts" marker or garbage.
const PTS_LIMIT: f64 = 1.0e17;
/// The error of a call on a context whose pipeline has stopped.
pub const STOPPED: &str = "pipeline stopped";

// The bed elements, in the order the encoder is given them and the object
// metadata describes them (see `oamd`), and the DCA speaker index each is
// taken from. The low frequency channel is first because the payload syntax
// puts it there.
// Element 0: LFE <- DCA Speaker 5 (LFE1)
// Element 1: L   <- DCA Speaker 1 (L)
// Element 2: R   <- DCA Speaker 2 (R)
// Element 3: C   <- DCA Speaker 0 (C)
// Element 4: Ls  <- DCA Speaker 3 (Ls/Lss)
// Element 5: Rs  <- DCA Speaker 4 (Rs/Rss)
// Element 6: Lrs <- DCA Speaker 7 (Lsr)
// Element 7: Rrs <- DCA Speaker 8 (Rsr)
const BED_SPEAKER_MAP: [usize; 8] = [5, 1, 2, 0, 3, 4, 7, 8];
// The element count of the stream and of its description are one number.
const _: () = assert!(CHANNELS == oamd::ELEMENTS);

/// Longest key accepted, in bytes.
pub const MAX_KEY_BYTES: usize = 128;

/// Parse a key written as hexadecimal text: an even number of hex digits,
/// nothing else but surrounding whitespace. Anything that is not clean hex
/// is an error and is never used as raw bytes.
#[allow(clippy::manual_is_multiple_of)] // `is_multiple_of` needs a newer compiler than the crate promises
pub fn parse_hex_key(text: &str) -> Result<Vec<u8>, &'static str> {
    let text = text.trim();
    if text.is_empty() {
        return Err("empty key");
    }
    if text.len() % 2 != 0 {
        return Err("odd number of hex digits");
    }
    if text.len() / 2 > MAX_KEY_BYTES {
        return Err("key too long");
    }
    let mut key = Vec::with_capacity(text.len() / 2);
    for pair in text.as_bytes().chunks(2) {
        let hi = (pair[0] as char).to_digit(16).ok_or("non-hex character")?;
        let lo = (pair[1] as char).to_digit(16).ok_or("non-hex character")?;
        key.push((hi * 16 + lo) as u8);
    }
    Ok(key)
}

/// Counters for logging and tests.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Stats {
    /// DTS frames decoded and encoded from real PCM.
    pub frames_decoded: u64,
    /// Of those, frames that had no lossless (XLL) data: the lossy core
    /// decode was carried instead.
    pub lossy_frames: u64,
    /// Frames replaced by silence of their length (decode error, XLL
    /// buffering, unsupported rate).
    pub silence_frames: u64,
    /// Times the parser skipped bytes to find the next frame.
    pub resyncs: u64,
    pub bytes_skipped: u64,
    /// Samples per channel handed to the encoder since the last reset.
    pub samples_fed: u64,
    /// Access units handed out by `pull_truehd`.
    pub units_out: u64,
    /// Whole units the encoder has been given, and the samples it holds that
    /// do not yet make one.
    pub units_encoded: u64,
    pub samples_carried: usize,
    /// Input bytes the parser is keeping for the next frame, and timestamps
    /// it has not matched to a frame yet.
    pub input_held: usize,
    pub pts_pending: usize,
    /// Panics caught and recovered from in the decode and encode threads.
    /// Each costs the audio in flight and nothing else.
    pub panics: u64,
    /// Units the encoder could not fit within the rate it declares, and the
    /// low bits cleared from the input in answer (see `Packer::watch_the_rate`).
    pub starved_units: u64,
    pub dropped_bits: u32,
}

fn valid_pts(pts: f64) -> bool {
    pts.is_finite() && (0.0..PTS_LIMIT).contains(&pts)
}

/// The bed elements, by their place in the planes (see `BED_SPEAKER_MAP`).
const LFE: usize = 0;
const L: usize = 1;
const R: usize = 2;
const C: usize = 3;
const LS: usize = 4;
const RS: usize = 5;
const LRS: usize = 6;
const RRS: usize = 7;

/// One row of a presentation over the sixteen elements: the gain of each
/// element listed, zero for the rest.
fn row(gains: &[(usize, f64)]) -> Vec<f64> {
    let mut out = vec![0.0; CHANNELS];
    for &(element, gain) in gains {
        out[element] = gain;
    }
    out
}

/// The four presentations of the stream from the rows of the three narrow
/// ones; the widest is the sixteen elements themselves.
fn presentations(
    two: Vec<Vec<f64>>,
    five: Vec<Vec<f64>>,
    seven: Vec<Vec<f64>>,
) -> Vec<Presentation> {
    let elements: Vec<Vec<f64>> = (0..CHANNELS).map(|c| row(&[(c, 1.0)])).collect();
    [(2usize, two), (6, five), (8, seven), (CHANNELS, elements)]
        .into_iter()
        .map(|(channels, rows)| Presentation { channels, rows })
        .collect()
}

/// The 2.0, 5.1 and 7.1 presentations of the sixteen-element stream as a
/// selection: each channel of each is one element of the bed at unity,
/// nothing mixed. The channels of a presentation are named in the order
/// L R C LFE Ls Rs Lrs Rrs, whatever the order of the elements they are
/// taken from (the low frequency element leads the stream), so a decoder that
/// stops at 2.0 plays L and R, one that stops at 5.1 plays L R C LFE Ls Rs,
/// and a 7.1 decoder gets the bed. The widest presentation is the sixteen
/// elements themselves.
///
/// A selection asks the encoder for nothing it can fail to give back, so it
/// is never refused (an interval of full-scale noise included), which is why
/// it is the set a stream carries when the folds are off and the set an
/// interval falls back to when a fold is refused. Without any presentation
/// stated the narrow ones would be the leading stored elements: the low
/// frequency channel and the left, labelled as stereo.
fn selection_presentations() -> Vec<Presentation> {
    let pick = |elements: &[usize]| -> Vec<Vec<f64>> {
        elements.iter().map(|&e| row(&[(e, 1.0)])).collect()
    };
    presentations(
        pick(&[L, R]),
        pick(&[L, R, C, LFE, LS, RS]),
        pick(&[L, R, C, LFE, LS, RS, LRS, RRS]),
    )
}

/// The same presentations as real folds of the sixteen elements: the 7.1 as
/// the selection, the 5.1 with the rear surrounds folded into the sides, and
/// the 2.0 with the centre and the sides folded into the fronts (the usual
/// coefficients, scaled so a fold cannot go past full scale). They carry the
/// whole bed to a decoder that stops short, at the price of an encoder that
/// can refuse them on a small share of intervals; see [`Options::folds`].
fn fold_presentations() -> Vec<Presentation> {
    let selected = selection_presentations();
    presentations(
        vec![
            row(&[(L, 0.5), (C, 0.354), (LS, 0.354), (LRS, 0.354)]),
            row(&[(R, 0.5), (C, 0.354), (RS, 0.354), (RRS, 0.354)]),
        ],
        vec![
            row(&[(L, 1.0)]),
            row(&[(R, 1.0)]),
            row(&[(C, 1.0)]),
            row(&[(LFE, 1.0)]),
            row(&[(LS, 0.707), (LRS, 0.707)]),
            row(&[(RS, 0.707), (RRS, 0.707)]),
        ],
        selected[2].rows.clone(),
    )
}

/// What a [`Context`] is built with beyond the sample rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
    /// One of [`EFFORT_REALTIME`], [`EFFORT_FAST`], [`EFFORT_FULL`].
    pub effort_level: u32,
    /// Whether the 2.0 and 5.1 presentations of the stream are real folds of
    /// the bed (see [`fold_presentations`]) rather than a selection of its
    /// channels (2.0 = L R, 5.1 = L R C LFE Ls Rs, 7.1 = the bed). Off by
    /// default: the encoder's fold cascade is refused on a small share of
    /// intervals of real content (0.2 to 6.3% measured). A refused interval
    /// falls back to the selection, so a decoder that stops at 2.0 or 5.1
    /// plays the named channels for those intervals instead of a fold, which
    /// is audible as a change of level; the sixteen-element presentation is
    /// unaffected either way.
    pub folds: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            effort_level: EFFORT_REALTIME,
            folds: false,
        }
    }
}

fn build_encoder(
    sample_rate: u32,
    options: Options,
    key: Option<&[u8]>,
) -> Result<MlpEncoder, hz_mlp::Unsupported> {
    let mut encoder = MlpEncoder::new(MlpConfig {
        sample_rate,
        channels: CHANNELS, // 16-element immersive TrueHD presentation
        bits: SampleBits::TwentyFour,
    })?;
    encoder.set_effort(match options.effort_level {
        EFFORT_REALTIME => Effort::Realtime,
        EFFORT_FAST => Effort::Fast,
        _ => Effort::Full,
    });
    // The narrow presentations are stated as exact selections of the bed, so
    // the encoder's decorrelation of the early channels must stay off: with
    // presentations stated it rewrites the rows the early substreams hand
    // back, and on correlated content (a real bed) the 2.0, 5.1 and 7.1
    // outputs then differ from the channels they name by up to hundreds of
    // least significant bits. The sixteen elements are exact either way. The
    // proper fix belongs in hz-mlp; until then the cost is a larger stream.
    encoder.set_decorrelation(false);
    // Always stated: a narrow decoder must never be handed the leading
    // elements, which start with the low frequency channel.
    if options.folds {
        encoder.set_presentations_with_fallback(&fold_presentations(), &selection_presentations());
    } else {
        encoder.set_presentations(&selection_presentations());
    }
    if let Some(key) = key {
        encoder.set_evolution_key(key);
    }
    Ok(encoder)
}

/// Hands PCM planes to the encoder and collects its output. The encoder cuts
/// the planes into access units itself and keeps a frame's remainder for the
/// next frame, so no sample is dropped or repeated. Lives on the encoder
/// thread.
struct Packer {
    sample_rate: u32,
    options: Options,
    key: Option<Vec<u8>>,
    encoder: MlpEncoder,
    /// The encoder was finished (or is in use) and must be rebuilt before
    /// the next stream starts.
    stale: bool,
    /// Samples per channel given to the encoder since the stream started.
    fed: u64,
    /// Zeroed planes for silence, grown to the longest asked for.
    zeros: Vec<i32>,
    /// Encoder output not yet handed to the output queue.
    staged: Vec<u8>,
    /// Units the encoder reported as impossible to place within the rate the
    /// stream declares, over all streams; and how many it had reported when
    /// this stream's encoder was last looked at.
    starved_total: u64,
    starved_seen: u64,
    /// Low bits cleared from every sample so content that needs more than
    /// the declared rate still fits it. Zero for all content that does.
    dropped_bits: u32,
    /// `fed` when the last bit was cleared: the encoder reports a unit about
    /// four intervals after it is fed, so a report sooner than that is about
    /// input from before the change.
    bits_changed_at: u64,
    masked: Vec<i32>,
}

impl Packer {
    fn new(sample_rate: u32, options: Options) -> Result<Self, hz_mlp::Unsupported> {
        Ok(Self {
            sample_rate,
            options,
            key: None,
            encoder: build_encoder(sample_rate, options, None)?,
            stale: false,
            fed: 0,
            zeros: Vec::new(),
            staged: Vec::with_capacity(1 << 17),
            starved_total: 0,
            starved_seen: 0,
            dropped_bits: 0,
            bits_changed_at: 0,
            masked: Vec::new(),
        })
    }

    /// A fresh encoder, so nothing of the stream before survives.
    /// A rebuild that fails leaves the packer stale, so it drops audio
    /// instead of pushing into an encoder that was finished or broken.
    fn restart(&mut self) {
        let used = self.stale || self.fed > 0;
        self.staged.clear();
        self.fed = 0;
        self.starved_seen = 0;
        self.dropped_bits = 0;
        self.bits_changed_at = 0;
        if used {
            match build_encoder(self.sample_rate, self.options, self.key.as_deref()) {
                Ok(encoder) => {
                    self.encoder = encoder;
                    self.stale = false;
                }
                Err(e) => {
                    log::error!("hz-stream: cannot rebuild the encoder: {e:?}");
                    self.stale = true;
                }
            }
        }
    }

    /// After a panic the encoder may have lost parts of itself: build a new one.
    fn recover(&mut self) {
        self.stale = true;
        self.restart();
    }

    fn set_key(&mut self, key: &[u8]) {
        self.wipe_key();
        self.key = Some(key.to_vec());
        self.encoder.set_evolution_key(key);
    }

    fn wipe_key(&mut self) {
        if let Some(key) = self.key.as_mut() {
            wipe(key);
        }
        self.key = None;
    }

    /// Carry an object audio metadata payload on the next unit. The encoder
    /// writes the Evolution frame round it, and signs it when it has a key.
    ///
    /// It describes the frame about to be fed, which starts as many samples
    /// into the unit it completes as the encoder holds already: the payload
    /// takes effect there, not at the start of the unit, so it does not
    /// lead its audio.
    fn set_evolution(&mut self, payload: &[u8]) {
        self.ensure_fresh();
        if self.stale {
            return;
        }
        let offset = self.carried() as u32;
        self.encoder.set_evolution_at(OAMD, payload, offset);
    }

    fn ensure_fresh(&mut self) {
        if self.stale {
            self.restart();
        }
    }

    /// `n` samples of sixteen planes laid end to end (the eight bed elements,
    /// the low frequency one first, then the eight extension waveforms).
    fn feed_planes(&mut self, planes: &[i32], n: usize) {
        self.ensure_fresh();
        if self.stale {
            return;
        }
        let bytes = if self.dropped_bits > 0 {
            let mask = !((1i32 << self.dropped_bits) - 1);
            self.masked.clear();
            self.masked.extend(planes.iter().map(|v| v & mask));
            self.encoder.push_planes(&self.masked, n)
        } else {
            self.encoder.push_planes(planes, n)
        };
        self.staged.extend_from_slice(&bytes);
        self.fed += n as u64;
        self.watch_the_rate();
    }

    /// Content that needs more than the declared peak rate (sixteen channels
    /// of 24-bit noise) produces units a decoder reports as arriving too
    /// fast. The encoder says so some intervals late; for every new such
    /// unit one more low bit is cleared from the input, up to four, which
    /// brings any content back under the rate for the rest of the stream.
    fn watch_the_rate(&mut self) {
        let starved = self.encoder.stats().starved;
        if starved > self.starved_seen {
            self.starved_total += starved - self.starved_seen;
            self.starved_seen = starved;
            if self.dropped_bits < MAX_DROPPED_BITS
                && self.fed >= self.bits_changed_at + REPORT_LAG_SAMPLES
            {
                self.dropped_bits += 1;
                self.bits_changed_at = self.fed;
                log::warn!(
                    "hz-stream: content exceeds the declared rate; clearing {} low bit(s) from now on",
                    self.dropped_bits
                );
            }
        }
    }

    fn silence(&mut self, n: usize) {
        if self.zeros.len() < CHANNELS * n {
            self.zeros.resize(CHANNELS * n, 0);
        }
        let zeros = std::mem::take(&mut self.zeros);
        self.feed_planes(&zeros[..CHANNELS * n], n);
        self.zeros = zeros;
    }

    /// Whole units given to the encoder, and the samples it holds that do not
    /// yet make one.
    fn units_in(&self) -> u64 {
        self.fed / BASE_FRAME_SIZE as u64
    }

    fn carried(&self) -> usize {
        (self.fed % BASE_FRAME_SIZE as u64) as usize
    }

    /// End the stream: the partial unit goes out as a short last unit (the
    /// format says how many of its samples to drop), then everything the
    /// encoder still holds. The next feed starts a new stream.
    fn finish(&mut self) {
        if self.stale || self.fed == 0 {
            return;
        }
        let tail = self.encoder.finish_carried();
        self.staged.extend_from_slice(&tail);
        self.fed = 0;
        self.stale = true;
    }
}

impl Drop for Packer {
    fn drop(&mut self) {
        self.wipe_key();
    }
}

fn wipe(key: &mut [u8]) {
    for b in key.iter_mut() {
        // SAFETY: b is a valid &mut u8.
        unsafe { std::ptr::write_volatile(b, 0) };
    }
}

// ---------------------------------------------------------------------------
// Threads
//
// The caller's thread only hands bytes over and pops units. A decode thread
// parses and decodes; an encode thread packs and encodes (its interval stages
// fork onto the shared bounded pool inside hz-mlp). Between them are bounded
// queues, so a slow encoder pushes back on the decoder and then on the caller
// rather than growing memory.

/// The encoder thread waits while this much output is queued and unpulled:
/// roughly half a second of audio at a typical Atmos rate, a little under a
/// second at the format's cap.
const OUTPUT_LIMIT: usize = 1 << 20;
/// Packets the caller may be ahead of the decoder by.
const INPUT_QUEUE: usize = 64;
/// Decoded frames in flight between the decoder and the encoder: one being
/// encoded and one being filled.
const FRAME_QUEUE: usize = 2;
/// How long a waiter sleeps before it re-checks, in case a wake-up was missed.
const POLL: Duration = Duration::from_millis(20);

/// What the encoder thread has produced and the caller has not pulled, and
/// what a pull needs to timestamp it.
#[derive(Default)]
struct Out {
    raw: Vec<u8>,
    raw_pos: usize,
    /// Frame timestamps, keyed on the stream sample position of the frame.
    /// Output units look themselves up on the same counter.
    pts_map: VecDeque<(u64, f64)>,
    units_out: u64,
    /// Units the encoder has written, whether or not they were pulled.
    units_written: u64,
    /// After an encoder restart, output unit `.0` is the audio at stream
    /// position `.1`: what the encoder lost is not counted in between.
    origin: (u64, u64),
    stats: Stats,
}

struct Shared {
    out: Mutex<Out>,
    /// Signalled when the caller pulls or the stream is reset.
    room: Condvar,
    /// Bumped by every reset; work from an older epoch is dropped.
    epoch: AtomicU64,
    /// A flush is outstanding: the output limit is not enforced, so the drain
    /// can finish while the caller waits for it.
    flushing: AtomicBool,
    /// A frame at a rate the encoder was not built for was seen.
    rate_refused: AtomicBool,
    /// A pipeline thread panicked.
    dead: AtomicBool,
    /// Tests make the next packet or frame panic.
    #[cfg(test)]
    inject_panic_decode: AtomicBool,
    #[cfg(test)]
    inject_panic_encode: AtomicBool,
    acked: Mutex<u64>,
    acked_cv: Condvar,
    /// Plane buffers on their way round: filled by the decode thread (or by
    /// `push_pcm`), read by the encode thread, then given back.
    planes: Pool<i32>,
    /// Packet buffers: filled by the caller, read by the decode thread.
    packets: Pool<u8>,
}

impl Shared {
    fn new() -> Self {
        Self {
            out: Mutex::new(Out::default()),
            room: Condvar::new(),
            epoch: AtomicU64::new(0),
            flushing: AtomicBool::new(false),
            rate_refused: AtomicBool::new(false),
            dead: AtomicBool::new(false),
            #[cfg(test)]
            inject_panic_decode: AtomicBool::new(false),
            #[cfg(test)]
            inject_panic_encode: AtomicBool::new(false),
            acked: Mutex::new(0),
            acked_cv: Condvar::new(),
            // Decoded PCM waits in the input queue, up to INPUT_QUEUE frames,
            // and the encoder hands them all back in a burst: a smaller keep
            // would drop the surplus and allocate it again.
            planes: Pool::new(INPUT_QUEUE + SPARE_FRAMES),
            packets: Pool::new(SPARE_PACKETS),
        }
    }

    fn out(&self) -> MutexGuard<'_, Out> {
        self.out.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn ack(&self, token: u64) {
        let mut acked = self.acked.lock().unwrap_or_else(|e| e.into_inner());
        *acked = (*acked).max(token);
        self.acked_cv.notify_all();
    }

    /// Wait until the pipeline has reached `token`. False if it died first.
    fn wait_ack(&self, token: u64) -> bool {
        let mut acked = self.acked.lock().unwrap_or_else(|e| e.into_inner());
        while *acked < token {
            if self.dead.load(Ordering::Acquire) {
                return false;
            }
            acked = self
                .acked_cv
                .wait_timeout(acked, POLL)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        true
    }

    fn mark_dead(&self) {
        self.dead.store(true, Ordering::Release);
        self.acked_cv.notify_all();
        self.room.notify_all();
    }
}

enum Input {
    Data {
        epoch: u64,
        bytes: Vec<u8>,
        pts: f64,
    },
    /// Decoded PCM: sixteen planes of `samples`, laid end to end.
    Pcm {
        epoch: u64,
        planes: Vec<i32>,
        samples: usize,
        pts: f64,
    },
    Flush {
        token: u64,
    },
    Reset {
        epoch: u64,
        token: u64,
    },
    Key(Vec<u8>),
    Barrier {
        token: u64,
    },
    Stop,
}

enum Work {
    Frame(FrameWork),
    Flush { token: u64 },
    Reset { token: u64 },
    Key(Vec<u8>),
    Barrier { token: u64 },
    Stop,
}

struct FrameWork {
    epoch: u64,
    /// The stream sample position of the frame's first sample.
    pos: u64,
    samples: usize,
    /// Sixteen planes of `samples`, laid end to end; empty for silence.
    planes: Vec<i32>,
    silence: bool,
    /// An EMDF payload to carry from this frame on.
    evolution: Option<Vec<u8>>,
}

struct Found {
    core: Range<usize>,
    exss: Range<usize>,
    /// PCM samples per channel the core header announces.
    samples: usize,
}

/// Fill `out[i]` for every `(i, slice)` the decoder offers.
fn gather<'a>(it: impl Iterator<Item = (usize, &'a [i32])>, out: &mut [&'a [i32]]) {
    for (i, slice) in it {
        if i < out.len() {
            out[i] = slice;
        }
    }
}

/// Sixteen planes laid end to end from the bed and extension slices, in the
/// order the object metadata describes them, written into `planes` (which keeps its room from frame to frame); a slice shorter
/// than `n` (or absent) contributes zeros.
fn lay_planes(planes: &mut Vec<i32>, bed: &[&[i32]; 8], ext: &[&[i32]; 8], n: usize) {
    planes.clear();
    planes.resize(CHANNELS * n, 0);
    for (c, src) in bed.iter().chain(ext.iter()).enumerate() {
        let len = src.len().min(n);
        planes[c * n..c * n + len].copy_from_slice(&src[..len]);
    }
}

/// The decode thread's state: bytes in, decoded frames out.
struct DecodeStage {
    sample_rate: u32,
    decoder: HdDecoder,
    current_frame: HdFrame,
    input_buf: Vec<u8>,
    /// Input bytes drained so far; the offset of `input_buf[0]` in the stream.
    consumed: u64,
    /// Pushed timestamps, keyed on the byte offset of their packet.
    pts_in: VecDeque<(u64, f64)>,
    stream_pos: u64,
    /// Samples of `stream_pos` that are padding after a flush, not audio.
    padding: u64,
    epoch: u64,
    author: oamd::Author,
    cleaner: bed::Cleaner,
    shared: Arc<Shared>,
    tx: SyncSender<Work>,
    stats: Stats,
}

impl DecodeStage {
    fn new(sample_rate: u32, shared: Arc<Shared>, tx: SyncSender<Work>) -> Self {
        let mut decoder = HdDecoder::new();
        // The encoder reads the integer taps; the f32 copy is never used.
        decoder.set_float_output(false);
        Self {
            sample_rate,
            decoder,
            current_frame: HdFrame::default(),
            input_buf: Vec::with_capacity(65536),
            consumed: 0,
            pts_in: VecDeque::new(),
            stream_pos: 0,
            padding: 0,
            epoch: 0,
            author: oamd::Author::new(),
            cleaner: bed::Cleaner::new(),
            shared,
            tx,
            stats: Stats::default(),
        }
    }

    fn run(&mut self, rx: Receiver<Input>) {
        while let Ok(input) = rx.recv() {
            match input {
                Input::Data { epoch, bytes, pts } => {
                    if epoch == self.shared.epoch.load(Ordering::Acquire) {
                        self.guard(|stage| stage.data(&bytes, pts));
                    }
                    self.shared.packets.give(bytes);
                }
                Input::Pcm {
                    epoch,
                    planes,
                    samples,
                    pts,
                } => {
                    if epoch != self.shared.epoch.load(Ordering::Acquire) {
                        self.shared.planes.give(planes);
                        continue;
                    }
                    if valid_pts(pts) {
                        self.shared.out().pts_map.push_back((self.stream_pos, pts));
                    }
                    self.stats.frames_decoded += 1;
                    let pos = self.stream_pos;
                    self.stream_pos += samples as u64;
                    self.send(Work::Frame(FrameWork {
                        epoch: self.epoch,
                        pos,
                        samples,
                        planes,
                        silence: false,
                        evolution: None,
                    }));
                }
                Input::Flush { token } => {
                    self.guard(|stage| {
                        stage.process_frames(true);
                    });
                    // The encoder pads the last unit to a whole one; count the
                    // padding so later timestamps are not late by it.
                    let part = (self.stream_pos % BASE_FRAME_SIZE as u64) as usize;
                    if part != 0 {
                        let pad = (BASE_FRAME_SIZE - part) as u64;
                        self.stream_pos += pad;
                        self.padding += pad;
                    }
                    self.send(Work::Flush { token });
                }
                Input::Reset { epoch, token } => {
                    self.decoder.reset();
                    self.input_buf.clear();
                    self.consumed = 0;
                    self.pts_in.clear();
                    self.stream_pos = 0;
                    self.padding = 0;
                    self.author.clear();
                    self.cleaner.reset();
                    self.epoch = epoch;
                    self.shared.out().pts_map.clear();
                    self.send(Work::Reset { token });
                }
                Input::Key(key) => self.send(Work::Key(key)),
                Input::Barrier { token } => self.send(Work::Barrier { token }),
                Input::Stop => break,
            }
            self.publish();
        }
        let _ = self.tx.send(Work::Stop);
    }

    fn send(&self, work: Work) {
        // A closed channel means the encoder thread is gone; its death is
        // already flagged and the caller finds out when it next waits.
        let _ = self.tx.send(work);
    }

    /// The decode-side counters, for [`Context::stats`].
    fn publish(&mut self) {
        let mut out = self.shared.out();
        out.stats.frames_decoded = self.stats.frames_decoded;
        out.stats.lossy_frames = self.stats.lossy_frames;
        out.stats.silence_frames = self.stats.silence_frames;
        out.stats.resyncs = self.stats.resyncs;
        out.stats.bytes_skipped = self.stats.bytes_skipped;
        out.stats.samples_fed = self.stream_pos - self.padding;
        out.stats.input_held = self.input_buf.len();
        out.stats.pts_pending = self.pts_in.len();
    }

    fn data(&mut self, data: &[u8], pts: f64) {
        if valid_pts(pts) {
            if self.pts_in.len() >= MAX_PTS_QUEUE {
                self.pts_in.pop_front();
            }
            self.pts_in
                .push_back((self.consumed + self.input_buf.len() as u64, pts));
        }
        self.input_buf.extend_from_slice(data);
        #[cfg(test)]
        if self
            .shared
            .inject_panic_decode
            .swap(false, Ordering::AcqRel)
        {
            panic!("injected decode panic");
        }
        let mut rate_ok = self.process_frames(false);
        // A buffer this large holds a false frame start that never completes:
        // drop its first byte and look again, keeping any good frames behind.
        while self.input_buf.len() > MAX_INPUT_BUFFER {
            self.discard(1);
            rate_ok &= self.process_frames(false);
        }
        if !rate_ok {
            self.shared.rate_refused.store(true, Ordering::Release);
        }
    }

    /// Run `f`; a panic costs the audio in flight and the stream state of
    /// the decoder, and the pipeline goes on with a fresh decoder.
    fn guard(&mut self, f: impl FnOnce(&mut Self)) {
        if catch_unwind(AssertUnwindSafe(|| f(self))).is_err() {
            log::error!("hz-stream: the decoder panicked, restarting it");
            self.decoder = HdDecoder::new();
            self.decoder.set_float_output(false);
            self.input_buf.clear();
            self.pts_in.clear();
            self.author.clear();
            self.cleaner.reset();
            self.shared.out().stats.panics += 1;
        }
    }

    fn discard(&mut self, n: usize) {
        if n > 0 {
            self.input_buf.drain(..n);
            self.consumed += n as u64;
            self.stats.resyncs += 1;
            self.stats.bytes_skipped += n as u64;
        }
    }

    /// Decode every complete frame in the input and hand it to the encoder.
    /// `last` says no more input will follow, so a trailing core-only frame
    /// is complete. Returns false if a frame had a rate the encoder was not
    /// built for.
    fn process_frames(&mut self, last: bool) -> bool {
        let mut rate_ok = true;
        while let Some(found) = self.find_next_frame(last) {
            let start_abs = self.consumed;
            let exss_bytes: &[u8] = if found.exss.is_empty() {
                &[]
            } else {
                &self.input_buf[found.exss.clone()]
            };
            let core_bytes = &self.input_buf[found.core.clone()];
            let decode_res =
                self.decoder
                    .decode_into(core_bytes, exss_bytes, &mut self.current_frame);
            let frame_end = if found.exss.is_empty() {
                found.core.end
            } else {
                found.exss.end
            };

            // The timestamp of the packet this frame starts in.
            let mut frame_pts = None;
            while let Some(&(off, pts)) = self.pts_in.front() {
                if off > start_abs {
                    break;
                }
                frame_pts = Some(pts);
                self.pts_in.pop_front();
            }
            if let Some(pts) = frame_pts {
                self.shared.out().pts_map.push_back((self.stream_pos, pts));
            }

            let fed = match decode_res {
                Ok(()) if self.current_frame.sample_rate == self.sample_rate => {
                    self.encode_frame(found.samples)
                }
                Ok(()) => {
                    rate_ok = false;
                    self.silence(found.samples);
                    found.samples
                }
                Err(e) => {
                    log::debug!("hz-stream: frame replaced by silence: {}", describe(&e));
                    self.silence(found.samples);
                    found.samples
                }
            };
            self.stream_pos += fed as u64;

            self.input_buf.drain(..frame_end);
            self.consumed += frame_end as u64;
        }
        rate_ok
    }

    fn silence(&mut self, samples: usize) {
        self.stats.silence_frames += 1;
        self.send(Work::Frame(FrameWork {
            epoch: self.epoch,
            pos: self.stream_pos,
            samples,
            planes: Vec::new(),
            silence: true,
            evolution: None,
        }));
    }

    /// Locate the next complete frame at the front of the input, skipping
    /// bytes that cannot start one. `None` means more input is needed. The
    /// skipped bytes are dropped once, however many false syncs they held.
    fn find_next_frame(&mut self, last: bool) -> Option<Found> {
        let mut at = 0;
        let found = self.scan_frame(&mut at, last);
        self.discard(at);
        found
    }

    /// The scan behind [`Self::find_next_frame`]: `at` ends at the first byte
    /// to keep, and a frame found there has its ranges counted from it.
    fn scan_frame(&self, at: &mut usize, last: bool) -> Option<Found> {
        let buf = &self.input_buf;
        loop {
            let Some(pos) = buf[*at..].windows(4).position(|w| w == CORE_SYNC) else {
                // Keep the tail: a sync word may be split across packets.
                *at = (*at).max(buf.len().saturating_sub(3));
                return None;
            };
            *at += pos;
            let rest = &buf[*at..];

            let info = match parse_header(rest) {
                Ok(info) => info,
                Err(ParseError::InsufficientData) => return None,
                Err(_) => {
                    *at += 1;
                    continue;
                }
            };
            if !(MIN_CORE_FRAME..=MAX_CORE_FRAME).contains(&info.frame_size) {
                *at += 1;
                continue;
            }
            let core_end = info.frame_size;
            if rest.len() < core_end {
                return None;
            }
            let samples = info.samples_per_channel();
            let core = 0..core_end;
            let core_only = Found {
                core: core.clone(),
                exss: core_end..core_end,
                samples,
            };
            if rest.len() < core_end + 4 {
                return if last { Some(core_only) } else { None };
            }
            if rest[core_end..core_end + 4] != SUBSTREAM_SYNC {
                return Some(core_only);
            }
            // The size field is read from the header alone, so a substream
            // cut by a packet boundary is waited for, not decoded without its
            // lossless and object data.
            match exss_substream_size(&rest[core_end..]) {
                Some(size) if size > MAX_EXSS_SIZE => return Some(core_only),
                Some(size) if rest.len() >= core_end + size => {
                    return Some(Found {
                        core,
                        exss: core_end..core_end + size,
                        samples,
                    });
                }
                // Too few bytes for the size field, or for the substream.
                _ if !last => return None,
                _ => return Some(core_only),
            }
        }
    }

    /// Hand the decoded frame to the encoder. Returns the samples per
    /// channel fed.
    fn encode_frame(&mut self, header_samples: usize) -> usize {
        let mut bed_slices: [&[i32]; 8] = [&[]; 8];
        let mut ext_slices: [&[i32]; dca::MAX_SOURCES] = [&[]; dca::MAX_SOURCES];
        let mut speakers: [&[i32]; 32] = [&[]; 32];

        // Choose the PCM source explicitly: the lossless XLL output when the
        // frame has it, else the integer output of the lossy core decode.
        let lossless = self.decoder.lossless_samples().next().is_some();
        let x_count;
        if lossless {
            gather(self.decoder.lossless_samples(), &mut speakers);
            gather(self.decoder.lossless_x_samples(), &mut ext_slices);
            x_count = self.decoder.lossless_x_samples().count();
        } else {
            gather(self.decoder.core_samples(), &mut speakers);
            gather(self.decoder.core_x_samples(), &mut ext_slices);
            x_count = self.decoder.core_x_samples().count();
            self.stats.lossy_frames += 1;
        }
        for c in 0..8 {
            bed_slices[c] = speakers[BED_SPEAKER_MAP[c]];
        }

        let frame_samples = bed_slices
            .iter()
            .chain(ext_slices.iter())
            .map(|s| s.len())
            .max()
            .filter(|&n| n > 0)
            .unwrap_or(header_samples);
        self.stats.frames_decoded += 1;

        // What the frame states of its extension waveforms, if anything: it
        // says what is folded into the bed and where each waveform belongs.
        let meta = oamd::parse(&self.current_frame, x_count);
        let mut planes = self.shared.planes.take();
        let slots = self.cleaner.clean(
            meta.as_ref(),
            &bed_slices,
            &ext_slices,
            frame_samples,
            &mut planes,
        );
        // The object metadata: always the bed, and the waveforms carried.
        // The payload is moved to the encoder as the writer made it.
        let evolution = self.author.describe(meta.as_ref(), &slots);
        self.send(Work::Frame(FrameWork {
            epoch: self.epoch,
            pos: self.stream_pos,
            samples: frame_samples,
            planes,
            silence: false,
            evolution,
        }));
        frame_samples
    }
}

/// The encode thread: frames in, TrueHD bytes into the output queue.
/// How many whole access units `bytes` holds (each starts with its length).
fn count_units(bytes: &[u8]) -> u64 {
    let (mut at, mut units) = (0, 0);
    while at + 2 <= bytes.len() {
        let len = ((((bytes[at] & 0x0f) as usize) << 8) | bytes[at + 1] as usize) * 2;
        if len < 4 || at + len > bytes.len() {
            break;
        }
        at += len;
        units += 1;
    }
    units
}

fn encode_thread(mut packer: Packer, rx: Receiver<Work>, shared: Arc<Shared>) {
    // Hand what the packer staged to the caller's queue, with the counters
    // the tests and the log read.
    fn publish(packer: &mut Packer, shared: &Shared) {
        let mut out = shared.out();
        out.units_written += count_units(&packer.staged);
        out.raw.extend_from_slice(&packer.staged);
        packer.staged.clear();
        out.stats.units_encoded = packer.units_in();
        out.stats.samples_carried = packer.carried();
        out.stats.starved_units = packer.starved_total;
        out.stats.dropped_bits = packer.dropped_bits;
    }

    // A panic in the encoder may have taken parts of it with it: build a new
    // one and go on, so the host does not have to reset to get audio again.
    fn recovered(packer: &mut Packer, shared: &Shared, resync: &mut bool) {
        *resync = true;
        log::error!("hz-stream: the encoder panicked, restarting it");
        packer.recover();
        shared.out().stats.panics += 1;
    }

    // Set by a restart after a panic: the next frame starts a new timeline.
    let mut resync = false;
    while let Ok(work) = rx.recv() {
        match work {
            Work::Frame(frame) => {
                // Wait for room in the output queue: the encoder, then the
                // decoder, then the caller, are held back by a caller that
                // has stopped pulling. A reset or a flush lets go at once.
                let mut out = shared.out();
                while out.raw.len() - out.raw_pos >= OUTPUT_LIMIT
                    && !shared.flushing.load(Ordering::Acquire)
                    && frame.epoch == shared.epoch.load(Ordering::Acquire)
                    && !shared.dead.load(Ordering::Acquire)
                {
                    out = shared
                        .room
                        .wait_timeout(out, POLL)
                        .unwrap_or_else(|e| e.into_inner())
                        .0;
                }
                drop(out);
                if frame.epoch != shared.epoch.load(Ordering::Acquire) {
                    shared.planes.give(frame.planes);
                    continue;
                }
                if resync {
                    resync = false;
                    let mut out = shared.out();
                    out.origin = (out.units_written, frame.pos);
                }
                if let Some(payload) = &frame.evolution {
                    packer.set_evolution(payload);
                }
                let fed = catch_unwind(AssertUnwindSafe(|| {
                    #[cfg(test)]
                    if shared.inject_panic_encode.swap(false, Ordering::AcqRel) {
                        panic!("injected encode panic");
                    }
                    if frame.silence {
                        packer.silence(frame.samples);
                    } else {
                        packer.feed_planes(&frame.planes, frame.samples);
                    }
                }));
                // The samples are in the encoder; the buffer goes round again.
                shared.planes.give(frame.planes);
                if fed.is_err() {
                    recovered(&mut packer, &shared, &mut resync);
                }
                publish(&mut packer, &shared);
            }
            Work::Flush { token } => {
                if catch_unwind(AssertUnwindSafe(|| packer.finish())).is_err() {
                    recovered(&mut packer, &shared, &mut resync);
                }
                publish(&mut packer, &shared);
                shared.ack(token);
            }
            Work::Reset { token } => {
                packer.restart();
                {
                    let mut out = shared.out();
                    out.raw.clear();
                    out.raw_pos = 0;
                    out.units_out = 0;
                    out.units_written = 0;
                    out.origin = (0, 0);
                    out.stats.units_encoded = 0;
                    out.stats.samples_carried = 0;
                }
                shared.ack(token);
            }
            Work::Key(key) => {
                packer.set_key(&key);
                let mut key = key;
                wipe(&mut key);
            }
            Work::Barrier { token } => shared.ack(token),
            Work::Stop => break,
        }
    }
}

pub struct Context {
    sample_rate: u32,
    shared: Arc<Shared>,
    tx: Option<SyncSender<Input>>,
    threads: Vec<JoinHandle<()>>,
    token: u64,
    /// Our copy of the key, wiped on drop, so a refused replacement can be
    /// told from an accepted one.
    key: Option<Vec<u8>>,
}

impl Context {
    pub fn new(sample_rate: u32, effort_level: u32) -> Result<Self, hz_mlp::Unsupported> {
        Self::with_options(
            sample_rate,
            Options {
                effort_level,
                ..Options::default()
            },
        )
    }

    pub fn with_options(sample_rate: u32, options: Options) -> Result<Self, hz_mlp::Unsupported> {
        // Before the first encoder builds the shared pool.
        cores::configure();
        let packer = Packer::new(sample_rate, options)?;
        let shared = Arc::new(Shared::new());
        let (tx_in, rx_in) = sync_channel::<Input>(INPUT_QUEUE);
        let (tx_work, rx_work) = sync_channel::<Work>(FRAME_QUEUE);

        let mut threads = Vec::with_capacity(2);
        let s = Arc::clone(&shared);
        let mut stage = DecodeStage::new(sample_rate, Arc::clone(&shared), tx_work);
        threads.push(
            std::thread::Builder::new()
                .name("hz-decode".into())
                .spawn(move || {
                    hz_mlp::confine_current_thread();
                    if catch_unwind(AssertUnwindSafe(|| stage.run(rx_in))).is_err() {
                        s.mark_dead();
                    }
                })
                .expect("spawn the decode thread"),
        );
        let s = Arc::clone(&shared);
        threads.push(
            std::thread::Builder::new()
                .name("hz-encode".into())
                .spawn(move || {
                    hz_mlp::confine_current_thread();
                    let s2 = Arc::clone(&s);
                    if catch_unwind(AssertUnwindSafe(|| encode_thread(packer, rx_work, s2)))
                        .is_err()
                    {
                        s.mark_dead();
                    }
                })
                .expect("spawn the encode thread"),
        );
        Ok(Self {
            sample_rate,
            shared,
            tx: Some(tx_in),
            threads,
            token: 0,
            key: None,
        })
    }

    fn send(&self, input: Input) -> Result<(), &'static str> {
        match &self.tx {
            Some(tx) if tx.send(input).is_ok() && !self.shared.dead.load(Ordering::Acquire) => {
                Ok(())
            }
            _ => Err(STOPPED),
        }
    }

    /// Send a command and wait for the encoder thread to reach it.
    fn command(&mut self, make: impl FnOnce(u64) -> Input) -> bool {
        self.token += 1;
        let token = self.token;
        self.send(make(token)).is_ok() && self.shared.wait_ack(token)
    }

    /// Wait until everything pushed so far has been decoded and encoded
    /// (what is then queued can be pulled). For tests and benchmarks; a
    /// player never needs it.
    pub fn sync(&mut self) -> bool {
        self.command(|token| Input::Barrier { token })
    }

    pub fn stats(&self) -> Stats {
        let out = self.shared.out();
        let mut stats = out.stats.clone();
        stats.units_out = out.units_out;
        stats
    }

    /// Drop everything of the stream so far, including audio the encoder was
    /// holding back, so none of it plays after a seek. Returns once both
    /// threads have dropped it.
    pub fn reset(&mut self) {
        let epoch = self.shared.epoch.fetch_add(1, Ordering::AcqRel) + 1;
        self.shared.room.notify_all();
        self.command(|token| Input::Reset { epoch, token });
    }

    pub fn set_key(&mut self, key: &[u8]) {
        if let Some(old) = self.key.as_mut() {
            wipe(old);
        }
        self.key = Some(key.to_vec());
        let _ = self.send(Input::Key(key.to_vec()));
    }

    /// [`Self::set_key`] from hex text; a malformed key is refused and the
    /// previous key (if any) stays.
    pub fn set_key_hex(&mut self, text: &str) -> Result<(), &'static str> {
        let mut key = parse_hex_key(text)?;
        self.set_key(&key);
        key.fill(0);
        Ok(())
    }

    /// Hand a packet over; decoding happens on the decode thread. A frame at
    /// a rate the encoder was not built for is carried as silence, and that is
    /// reported by the next call, since the frame is not decoded yet when this
    /// one returns.
    pub fn push_dts(&mut self, data: &[u8], pts: f64) -> Result<usize, &'static str> {
        let epoch = self.shared.epoch.load(Ordering::Acquire);
        // The copy goes into a buffer that comes back from the decode thread,
        // so a packet costs no allocation once the loop is warm.
        let mut bytes = self.shared.packets.take();
        bytes.extend_from_slice(data);
        self.send(Input::Data { epoch, bytes, pts })?;
        if self.shared.rate_refused.swap(false, Ordering::AcqRel) {
            Err("unsupported sample rate")
        } else {
            Ok(data.len())
        }
    }

    /// Feed already-decoded PCM, the eight bed planes then the eight extension
    /// planes. For tests and benchmarks that need a sixteen-channel stream
    /// without a DTS:X fixture; the timestamp rules are those of
    /// [`Self::push_dts`], keyed on the same sample counter.
    pub fn push_pcm(
        &mut self,
        bed: &[&[i32]; 8],
        ext: &[&[i32]; 8],
        pts: f64,
    ) -> Result<(), &'static str> {
        let samples = bed
            .iter()
            .chain(ext.iter())
            .map(|s| s.len())
            .max()
            .unwrap_or(0);
        let mut planes = self.shared.planes.take();
        lay_planes(&mut planes, bed, ext, samples);
        let epoch = self.shared.epoch.load(Ordering::Acquire);
        self.send(Input::Pcm {
            epoch,
            planes,
            samples,
            pts,
        })
    }

    /// The time of output unit `unit`, from the same sample counter the
    /// frame timestamps are keyed on.
    fn unit_pts(&self, out: &mut Out, unit: u64) -> f64 {
        let (origin_unit, origin_pos) = out.origin;
        let pos = origin_pos + unit.saturating_sub(origin_unit) * BASE_FRAME_SIZE as u64;
        while out.pts_map.len() > 1 && out.pts_map[1].0 <= pos {
            out.pts_map.pop_front();
        }
        match out.pts_map.front() {
            Some(&(p0, t0)) => t0 + (pos as f64 - p0 as f64) * 1.0e6 / self.sample_rate as f64,
            None => f64::NAN,
        }
    }

    /// Copy the next TrueHD access unit into `out_buf`. Returns its length;
    /// 0 when no whole unit is ready; `Err(needed)` if `out_buf` is too small.
    /// Never waits.
    pub fn pull_truehd(&mut self, out_buf: &mut [u8], out_pts: &mut f64) -> Result<usize, usize> {
        let shared = Arc::clone(&self.shared);
        let mut out = shared.out();
        let avail = &out.raw[out.raw_pos..];
        if avail.len() < 2 {
            return Ok(0);
        }
        // The unit's first 16 bits: a 4-bit check nibble, then its length in
        // 16-bit words.
        let len = ((((avail[0] & 0x0f) as usize) << 8) | avail[1] as usize) * 2;
        if len < 4 || len > avail.len() {
            if len < 4 {
                log::warn!("hz-stream: corrupt TrueHD unit header, dropping queued output");
                out.raw.clear();
                out.raw_pos = 0;
            }
            return Ok(0);
        }
        if out_buf.len() < len {
            return Err(len);
        }
        let pos = out.raw_pos;
        out_buf[..len].copy_from_slice(&out.raw[pos..pos + len]);
        out.raw_pos += len;
        if out.raw_pos == out.raw.len() {
            out.raw.clear();
            out.raw_pos = 0;
        } else if out.raw_pos > (1 << 16) {
            let drained = out.raw_pos;
            out.raw.drain(..drained);
            out.raw_pos = 0;
        }
        let unit = out.units_out;
        *out_pts = self.unit_pts(&mut out, unit);
        out.units_out += 1;
        drop(out);
        shared.room.notify_one();
        Ok(len)
    }

    /// Bytes queued and not yet pulled.
    pub fn queued(&self) -> usize {
        let out = self.shared.out();
        out.raw.len() - out.raw_pos
    }

    /// End of stream (or before a seek that will `reset`): decode what is
    /// buffered, pad the last partial unit and drain the encoder. Waits for
    /// that, and returns the bytes now queued for pulling.
    pub fn flush(&mut self) -> usize {
        self.shared.flushing.store(true, Ordering::Release);
        self.shared.room.notify_all();
        self.command(|token| Input::Flush { token });
        self.shared.flushing.store(false, Ordering::Release);
        self.queued()
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if let Some(key) = self.key.as_mut() {
            wipe(key);
        }
        // Let go of a stalled encoder, then stop both threads.
        self.shared.epoch.fetch_add(1, Ordering::AcqRel);
        self.shared.flushing.store(true, Ordering::Release);
        self.shared.room.notify_all();
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(Input::Stop);
        }
        for handle in self.threads.drain(..) {
            let _ = handle.join();
        }
    }
}

fn describe(e: &HdError) -> &'static str {
    match e {
        HdError::Header(_) => "bad header",
        HdError::Core => "core decode",
        HdError::Exss => "substream",
        HdError::Xll(_) => "lossless extension",
        HdError::Pending => "lossless data still buffering",
    }
}
