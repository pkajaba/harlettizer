//! The encoder: samples in, access units out.

use crate::filter::Effort;
use crate::filter::{self, Coding, MAX_IIR_ORDER};
use crate::format::MAX_SUBSTREAMS;
use crate::format::{Coded, Config, DynamicRange, Unsupported};
use crate::frame::{self, Block, Unit};
use crate::matrix::{self, Primitive};

/// The end-of-stream marker, which is also how a final short unit says how
/// many of its samples are padding.
const END_OF_STREAM: u32 = 0xd234;

/// How often the stream restarts, in access units.
///
/// A restart is a point a decoder can join at, and it costs: a header per
/// substream, the channel assignment, the matrices restated, and the filters
/// and their history thrown away. The format caps the gap at 128 — a decoder
/// says so by name — and a reference stream sits right at the cap, so this
/// does too.
///
/// It was 16 for a long while, when moving it to 128 was worth 1.5 % and did
/// not look like the place to spend effort. After the filters, the dead bits
/// and the matrix search it is worth **5.3 %**, because the stream around it
/// got 40 % smaller and the framing did not. A lever measured once and put
/// down is worth picking up again when everything around it has changed.
const RESTART_INTERVAL: u64 = 128;

/// The codec's domain: twenty-four bits, which a decoder enforces on every
/// stored sample it reconstructs — the reference reports one past them as a
/// saturated recorrelator and refuses the stream. Under restart sync word C
/// the filters run in thirty-two bits and the bound is the whole word; see
/// [`Encoder::domain_of`].
const DOMAIN: i32 = 1 << 23;

/// How much of the decoder's input buffer a stream gives itself, in samples.
///
/// An access unit's bytes are delivered over the gap between its own arrival
/// time and the next one's, and a decoder refuses a unit that would arrive
/// faster than the peak the stream declared. So a unit larger than the average
/// has to be given a longer gap — and the time for it is borrowed against the
/// clock, then repaid by the units that are smaller than average.
///
/// This is how much may be borrowed at once. Eight access units is under seven
/// milliseconds at 48 kHz, an order inside the seventy-five the format allows,
/// and more than the reference stream ever uses: measured over its first 25876
/// units, its buffer swings between −60 and +113 samples of where it started.
const INPUT_RESERVE: usize = 8;

/// Samples in the first block of an access unit that restarts the decoder.
///
/// A restart header is where a decoder may join the stream, so nothing after
/// it may depend on what came before — and the filter state would. So the
/// block that follows one is coded with no prediction at all, which both
/// refills that state from its own data and sidesteps a disagreement between
/// the two decoders this project checks against: FFmpeg's carries its state
/// across a restart, `truehdd` clears it. A stream whose restart block depends
/// on that state is one only one of them reconstructs, and for a while this
/// encoder wrote exactly that: FFmpeg decoded it bit for bit and `truehdd`
/// began a slowly decaying error at every restart but the first, because the
/// reconstruction is recursive and a wrong initial state rings out through it.
///
/// Eight samples, because eight is the longest filter the format has — and
/// because the format itself says so: after a restart header a decoder's block
/// size is eight and its filter orders are zero, which is this block exactly.
///
/// It costs what it costs: coding eight samples of every restart interval
/// without prediction. Coding the whole unit that way instead — the obvious
/// simpler fix, and the one that was measured first — cost 2.8 % of the stream
/// on real material where this costs about half a per cent.
pub(crate) const RESTART_BLOCK: usize = crate::lpc::MAX_ORDER;

/// A sliding window of a channel's recent past, without moving it every unit.
///
/// Appending to a `Vec` and draining its front moves the whole history on every
/// access unit — at sixteen channels and two thousand samples of history that
/// is tens of megabytes a second of memory traffic, for a window that only
/// slid by forty. This holds the window inside a buffer twice its size and
/// slides an offset instead, compacting once every `keep / unit` units.
///
/// The window is still one contiguous slice in time order and still exactly as
/// long, so everything reading it — the fit context, the filter state, the
/// matrix weight — sees the same samples in the same order, down to the
/// rounding of the sums made over them.
#[derive(Debug, Clone)]
struct History {
    buffer: Vec<i32>,
    /// Where the window starts inside it.
    start: usize,
    /// How much of the past to keep.
    keep: usize,
}

impl History {
    fn new(keep: usize, unit: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(2 * keep + unit),
            start: 0,
            keep,
        }
    }

    fn push(&mut self, samples: &[i32]) {
        self.buffer.extend_from_slice(samples);
        if self.buffer.len() - self.start > self.keep {
            self.start = self.buffer.len() - self.keep;
        }
        // Only when the dead space in front has grown to the window's own
        // length, which is once every `keep / unit` units rather than every
        // one of them.
        if self.start >= self.keep {
            self.buffer.copy_within(self.start.., 0);
            self.buffer.truncate(self.keep);
            self.start = 0;
        }
    }

    fn window(&self) -> &[i32] {
        &self.buffer[self.start..]
    }
}

/// What judging a destination's candidates came to.
#[derive(Debug, Default)]
struct Judged {
    /// The best of them, or none where coding the channel alone is cheaper.
    best: Option<Primitive>,
    /// Candidates refused because what they left did not fit the codec's
    /// domain.
    refused: u64,
}

/// One source set a destination might be coded against, and the weights that
/// fit it.
#[derive(Debug, Clone, Default)]
struct Candidate {
    sources: Vec<usize>,
    /// The weight each source carries at the *start* of the interval.
    weights: Vec<i32>,
    /// And how far it moves per access unit, measured across the interval
    /// rather than extrapolated from the one before it. Zero where the weight
    /// stands still or where the substream cannot say a step.
    steps: Vec<i32>,
}

impl Candidate {
    /// Whether this is the candidate a matrix was built from.
    ///
    /// By its sources and their weights, which is what identifies it: two
    /// candidates for one destination differ in how many sources they use.
    fn matches(&self, matrix: &Primitive) -> bool {
        self.sources
            .iter()
            .zip(&self.weights)
            .all(|(source, weight)| matrix.coefficients[*source] == *weight)
            && (0..matrix.coefficients.len()).all(|source| {
                source == matrix.dest
                    || matrix.coefficients[source] == 0
                    || self.sources.contains(&source)
            })
    }
}

/// How many samples of an interval the source *ranking* looks at.
///
/// The ranking is a comparison between candidate sources and a correlation
/// estimated on a couple of thousand samples of a passage is the same
/// correlation as one estimated on twenty thousand. The weights that go on the
/// wire are fitted on all of it.
const RANK_SAMPLES: usize = 2048;

/// Segments the interval is fitted over to find where a weight is going.
///
/// The format lets a matrix carry a step, so the coefficient it states is the
/// one at the interval's start and not the interval's average — and the two
/// differ by half the drift, which on material whose correlation moves is the
/// whole of the gain. Four segments is enough to put a line through and short
/// enough that each still holds a thousand samples.
const MATRIX_SEGMENTS: usize = 4;

/// One access unit's samples, waiting for its interval to fill.
///
/// The payload travels with them. What rides in a unit's extra data describes
/// *that* unit, and a caller sets it before pushing the samples it belongs to —
/// so holding an interval and reading the payload at write time would give
/// every unit of the interval the last one's.
#[derive(Debug, Default)]
struct Held {
    /// Interleaved, always a whole unit wide; a short final unit is padded
    /// with the zeroes the format says it carries.
    samples: Vec<i32>,
    /// How many of them are real.
    frames: usize,
    padding: usize,
    /// The Evolution payload this unit was given, and under which identifier.
    /// `None` is a unit that was given none, which is not the same as one
    /// given no bytes.
    evolution: Option<u32>,
    payload: Vec<u8>,
    /// The sample of the unit the payload takes effect at.
    evolution_offset: u32,
}

/// What one channel's search decided.
#[derive(Debug, Clone, Copy)]
struct Decided {
    coding: Coding,
    write_fir: bool,
    write_iir: bool,
    write_params: bool,
}

impl Default for Decided {
    fn default() -> Self {
        Self {
            coding: Coding {
                filter: crate::filter::Filter::default(),
                codebook: crate::huffman::NONE,
                huff_lsbs: crate::frame::CODEC_BITS,
            },
            write_fir: true,
            write_iir: false,
            write_params: true,
        }
    }
}

/// Where the per-channel search runs.
///
/// A pool of its own, sized to the channel count, rather than the global one.
/// A stream has between one and sixteen channels and a machine may have far
/// more cores than that: handing six independent searches to a pool of
/// thirty-two costs more in dispatch and stolen cache lines than the extra
/// threads can win back. Measured on a 5.1 programme at full effort — 3.34 s
/// through a thirty-two thread pool against 1.86 s through a six-thread one,
/// for exactly the same work.
///
/// Without the feature it is a unit, and every `install` below is the closure
/// called where it stands.
#[cfg(feature = "parallel")]
#[derive(Debug)]
struct Threads(Option<rayon::ThreadPool>);

#[cfg(not(feature = "parallel"))]
#[derive(Debug)]
struct Threads;

impl Threads {
    #[cfg(feature = "parallel")]
    fn new(channels: usize) -> Self {
        // One channel has nothing to spread, and a pool that cannot be built
        // is not a failure: the search runs where it stands.
        if channels < 2 {
            return Self(None);
        }
        let cores = std::thread::available_parallelism().map_or(channels, |n| n.get());
        Self(
            rayon::ThreadPoolBuilder::new()
                .num_threads(cores)
                .build()
                .ok(),
        )
    }

    /// A pool of exactly `threads` threads.
    #[cfg(feature = "parallel")]
    fn exactly(threads: usize) -> Self {
        Self(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads.max(1))
                .build()
                .ok(),
        )
    }

    #[cfg(not(feature = "parallel"))]
    fn exactly(_threads: usize) -> Self {
        Self
    }

    #[cfg(not(feature = "parallel"))]
    fn new(_channels: usize) -> Self {
        Self
    }

    /// `f` over every item, on the pool where there is one, in the items'
    /// order either way — so that whatever is decided from the results is
    /// decided the same as it would be one at a time.
    #[cfg(feature = "parallel")]
    fn map<T: Sync, R: Send>(&self, items: &[T], f: impl Fn(&T) -> R + Sync + Send) -> Vec<R> {
        use rayon::prelude::*;
        match &self.0 {
            Some(pool) => pool.install(|| items.par_iter().map(f).collect()),
            None => items.iter().map(f).collect(),
        }
    }

    #[cfg(not(feature = "parallel"))]
    fn map<T, R>(&self, items: &[T], f: impl Fn(&T) -> R) -> Vec<R> {
        items.iter().map(f).collect()
    }
}

/// What one access unit says about itself, decided before it is written.
///
/// A parameter object rather than eight arguments: they are all decisions the
/// steps above made, and passing them positionally is how one gets swapped for
/// another.
struct Written {
    /// Whether this unit restarts the decoder.
    restart: bool,
    /// Where the unpredicted first block ends, or zero when there is none.
    split: usize,
    write_shift: bool,
    write_matrix: bool,
    /// The check the restart header certifies, per substream.
    certified: [u8; MAX_SUBSTREAMS],
    /// How many of this unit's samples are padding.
    padding: usize,
}

/// What the first pass over a unit measured: the check each substream will
/// certify, and each channel's dead bits — or none for a channel silent all
/// unit, which takes the shift in force.
struct Measured {
    check: [u32; MAX_SUBSTREAMS],
    wasted: [Option<u8>; crate::format::MAX_CHANNELS],
}

/// The lossless check of one unit of an early substream's presentation: its
/// rows run over the channels as they stand, the way a decoder runs them,
/// then each channel shifted by what the substream states for it.
///
/// The decoder's arithmetic and nothing else: each row assigns its
/// destination from every channel the substream reads, its own included, as
/// one accumulation shifted down once; the rows run in order, each reading
/// what the ones before it left; and the check folds each output in at its
/// matrix channel's own index, which wraps at eight.
fn presentation_check<P: AsRef<[i32]>>(
    planes: &[P],
    span: usize,
    rows: &[Primitive],
    shifts: Option<&Vec<u8>>,
    frames: usize,
) -> u32 {
    let mut check = 0u32;
    let mut buffer = [0i64; crate::matrix::MAX_SOURCES];
    for frame in 0..frames {
        for (channel, slot) in buffer.iter_mut().enumerate().take(span + 1) {
            *slot = i64::from(planes[channel].as_ref()[frame]);
        }
        for row in rows {
            let mut accumulator = 0i64;
            for (channel, sample) in buffer.iter().enumerate().take(span + 1) {
                accumulator += sample * i64::from(row.coefficients[channel]);
            }
            buffer[row.dest] = accumulator >> crate::matrix::FRACTION;
        }
        for (channel, sample) in buffer.iter().enumerate().take(span + 1) {
            let shift = shifts.and_then(|s| s.get(channel)).copied().unwrap_or(0);
            let out = (*sample as i32) << shift;
            check ^= ((out as u32) & 0x00ff_ffff) << (channel & 7);
        }
    }
    check
}

/// Why an interval's folds could not be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Refusal {
    /// No cascade hosts the rows, or a presentation cannot be written over
    /// what the channels would hold.
    Unwritable,
    /// The cascade would store more than the codec's domain holds.
    OverTheDomain,
}

#[derive(Debug)]
/// What a build produced, all of which takes effect together.
struct Built {
    steps: Vec<Primitive>,
    rows: Vec<Vec<Primitive>>,
    assignment: Vec<Vec<u8>>,
    /// What each early substream states as its own output shift.
    stated: Vec<Vec<u8>>,
    /// Which element each internal channel carries under the cascade.
    element_of: Vec<usize>,
}

#[derive(Debug)]
/// A TrueHD encoder.
///
/// Feed it whole access units with [`Encoder::push`] and finish with
/// [`Encoder::finish`], which pads the last one and marks how much of it to
/// throw away. One unit in [`RESTART_INTERVAL`] restarts the decoder and is
/// independently decodable; the rest carry only what changed, against the
/// model of what the decoder is holding in [`crate::model`].
pub struct Encoder {
    coded: Coded,
    sample_rate: u32,
    channels: usize,
    shift: u32,
    /// What has already been encoded, per channel, most recent last. Its tail
    /// is the filter state a decoder carries across access units; the rest is
    /// context for choosing the next filter.
    past: Vec<History>,
    /// The matrices this unit carries, in the order a decoder applies them.
    ///
    /// One list, not one per substream, because a full decode applies only the
    /// *last* substream's matrices — so that substream has to declare every
    /// one of them, whatever channel it writes. See [`crate::frame`] for what
    /// each substream then says.
    matrices: Vec<Primitive>,
    /// What each substream before the last declares instead: the rows that
    /// compute its presentation from the internal channels. Empty until a
    /// caller states one, which leaves the narrow presentations as the leading
    /// channels — see [`Self::set_presentation`].
    presentations: Vec<Vec<Primitive>>,
    /// The steps that turn the stored channels back into the elements.
    ///
    /// Fixed for a whole stream, unlike the coding matrices, because it comes
    /// from the geometry of the folds and not from the audio. The encoder
    /// *undoes* them before it codes anything, so what it codes is the
    /// hierarchy of presentations rather than the elements; the last substream
    /// declares them after its coding matrices, so a full decode puts the
    /// elements back.
    arrangement: Vec<Primitive>,
    /// The next arrangement and the presentations that go with it, waiting for
    /// a restart to take effect.
    ///
    /// 🔴 They have to arrive together and at a restart. The declared matrices
    /// stand for a whole interval, so an arrangement adopted in the middle of
    /// one would code samples the header does not describe; and presentations
    /// written over channels a *different* arrangement left are presentations
    /// of nothing. One field, one adoption point.
    /// What each presentation is, over the elements, waiting for a restart.
    ///
    /// The rows and not the matrices, because the matrices cannot be worked
    /// out until the dead bits are: a channel shifted down by three and one
    /// shifted down by four are not in the same units, and a row that mixes
    /// them has to say so. So the caller states the *fold* and the encoder
    /// states the matrices, at the restart where it settles the shifts.
    waiting: Option<Vec<crate::hierarchy::Presentation>>,
    /// And what is in force, which is rebuilt whenever the shifts move.
    wants: Vec<crate::hierarchy::Presentation>,
    /// What each early substream states as its own output shift, which is not
    /// what the encoder shifted by. See where it is chosen.
    presentation_shift: Vec<Vec<u8>>,
    /// The shifts this interval holds for every one of its units, where they
    /// are held rather than taken again each time.
    ///
    /// A restart header states the matrices once for a whole interval, so the
    /// units of that interval have to be in the units those matrices were
    /// written for. The smallest any unit of it offers, since it has to be a
    /// shift every one of them really has.
    interval_shift: Option<Vec<u8>>,
    /// Which output channel each substream's matrix channels are, where a
    /// presentation needs them permuted.
    assignment: Vec<Vec<u8>>,
    /// Which element each internal channel carries: `permutation[channel]`.
    ///
    /// The identity until a hierarchy of presentations is built, and then
    /// what [`crate::arrange::order_elements`] chose — so that each fold row
    /// is hosted on an element that carries a large share of it, inside the
    /// channels its presentation reads. Stored in the elements' own order, a
    /// low frequency bed at element 0 sits in the stereo pair's channels
    /// carrying nothing of the stereo fold, and the row hosted there is stored
    /// at sixteen times its size. The last substream's channel assignment
    /// puts the elements back in order on the way out, so nothing outside the
    /// codec — the object metadata in particular — knows.
    permutation: Vec<usize>,
    /// Per channel, the dead low bits taken off this unit, which the decoder
    /// puts back after everything else.
    output_shift: Vec<u8>,
    /// The second filter each channel carries through the current restart
    /// interval, decided at its start on the interval it is about to meet — an
    /// index into the designs, or none. See [`crate::filter::pick_second`].
    second_mode: Vec<Option<usize>>,
    /// How hard that decision searches.
    effort: Effort,
    /// What each channel's search decided, this unit and in the unpredicted
    /// block of a restarting one. Held rather than returned so that spreading
    /// Where the per-channel search runs.
    threads: std::sync::Arc<Threads>,
    /// The interval that is complete and not yet written — see
    /// [`Encoder::advance`].
    deferred: Option<Deferred>,
    /// What writes the stream: everything that changes as units are written,
    /// and the interval being written. Out of the encoder while it writes an
    /// interval beside the deciding of the next — see [`Encoder::advance`].
    writer: Option<Writer>,
    /// The Evolution payload the next unit pushed is given, and where in the
    /// unit it takes effect; [`Encoder::hold`] keeps it with the unit.
    evolution: Vec<u8>,
    evolution_id: u32,
    evolution_offset: u32,
    evolution_pending: bool,
    /// The interval that is decided and not yet written.
    decided: Option<Record>,
    /// What decides each block of an interval, channel by channel, and its
    /// buffers. Out of the encoder while it works on one interval beside the
    /// preparing of the next — see [`IntervalDecider`].
    back: Option<IntervalDecider>,
    /// The interval that is prepared and not yet decided block by block.
    fronted: Option<Front>,
    /// How many intervals have been decided, which is what a second filter
    /// decided every other restart counts.
    intervals_decided: u64,
    /// The interval prepared for coding: per channel, every unit's samples
    /// with the dead bits off and the matrices applied, one after another.
    prepared: Vec<Vec<i32>>,
    /// The dead-bit shift each channel of each unit was taken at, and the
    /// check each unit folds to.
    prepared_shifts: Vec<u8>,
    prepared_checks: Vec<[u32; MAX_SUBSTREAMS]>,
    /// What the restart header opening the prepared interval declares.
    max_shift: u8,
    max_output_bits: u8,
    /// The interval being held, and the buffers its units came back in.
    pending: Vec<Held>,
    spare: Vec<Held>,
    /// A partial access unit [`Encoder::push_planes`] has taken and not yet
    /// completed, interleaved, and how many frames of it there are.
    carry: Vec<i32>,
    carry_frames: usize,
    /// The held interval's units, each one's channels side by side, while
    /// they are prepared: one chunk per unit is what lets the units go across
    /// the pool. Kept between intervals rather than allocated for each.
    unit_scratch: Vec<i32>,
    /// Per destination channel, the source sets the interval suggests, in
    /// increasing size, with the weights that fit them. Decided once an
    /// interval; costed by the unit that opens it.
    candidates: Vec<Vec<Candidate>>,
    /// The dynamic range each substream currently asks for, and what was last
    /// written for it. A substream that has never been given one carries no
    /// second directory word at all, which is two bytes an access unit saved
    /// over saying "still nothing".
    dynamic_range: [Option<DynamicRange>; MAX_SUBSTREAMS],
    /// Whether a hierarchy's early channels are decorrelated — see
    /// [`Encoder::decorrelate`]. On unless turned off.
    decorrelation: bool,
    /// Access units written since the last restart.
    since_restart: u64,
    stats: Stats,
    /// Wall time spent in each phase of an interval, kept only when
    /// `HZ_TIME` is set and printed when the stream is finished.
    phase_times: [std::time::Duration; 8],
    timed: bool,
}

/// What the encoder chose, summed over a stream.
///
/// Not diagnostics for their own sake: the only two questions worth asking of
/// a lossless encoder are whether it is still exact and where its bits went,
/// and the second one is unanswerable without this.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Stats {
    /// Blocks coded at each filter order, zero included.
    pub orders: [u64; crate::lpc::MAX_ORDER + 1],
    /// Blocks coded at each second-filter order, zero included.
    pub iir_orders: [u64; crate::filter::MAX_IIR_ORDER + 1],
    /// Channel-blocks that restated their first filter, and the total. A
    /// filter is kept until it stops paying, so the ratio is how long one
    /// lives — which is what a block's share of its description ought to be.
    pub filters_restated: u64,
    pub filter_blocks: u64,
    /// Channel-blocks that said anything about their coding at all — a
    /// filter, a codebook or a width that differs from what the decoder
    /// holds. The rest cost one bit each.
    pub params_restated: u64,
    /// Blocks coded with each codebook, "none" included.
    pub codebooks: [u64; crate::huffman::MAX_CODEBOOK as usize + 1],
    /// Access units that carried a matrix, and units in total.
    pub matrixed_units: u64,
    /// Matrices given a step to follow, because their weight was drifting.
    /// Only the last substream of an immersive stream can say this, and only
    /// for a channel no earlier one describes.
    pub aimed_matrices: u64,
    /// Matrices refused because what they left did not fit the codec's
    /// domain: candidates turned away on the weights they were judged with,
    /// and decided matrices dropped on what was really stored — see
    /// [`Encoder::judge_matrix`] and [`Encoder::prepare`].
    pub matrices_refused: u64,
    /// Restart intervals that were asked for folds — see
    /// [`Encoder::set_presentations`] — and how many of them carry them. The
    /// rest were refused and carry the leading elements instead, so that the
    /// stream is written without folds rather than written wrong.
    pub folds_asked: u64,
    pub folds_carried: u64,
    /// Of the refused, those whose cascade would have left the codec's
    /// twenty-four bits: the programme mixed too near full scale for what the
    /// cascade stores. Every other refusal is a set of rows no cascade could
    /// host or write.
    pub folds_over_the_domain: u64,
    pub units: u64,
    /// Total residual bits, and how many residuals, so the mean is
    /// recoverable.
    pub residual_bits: u64,
    pub residuals: u64,
    /// Bytes of framing: everything that is not a residual.
    pub framing_bytes: u64,
    pub total_bytes: u64,
    /// Access units the arrival schedule could not place: ones whose bytes
    /// cannot reach a decoder in time however the clock is arranged, because
    /// the stream needs more than it declared on *average* and not merely in
    /// one unit. Zero in a stream a decoder will accept.
    pub starved: u64,
    /// How empty the decoder's input buffer ever got, in samples. Negative is
    /// the same failure `starved` counts; a small positive number is a stream
    /// that spent most of what it had.
    pub lowest_advance: i64,
    /// The largest access unit written, in bytes.
    ///
    /// Which is the stream's real peak rate, and the one number the declared
    /// peak in the major sync has to be at or above. At sixteen channels the
    /// format's ceiling is below the uncompressed rate, so this stops being a
    /// formality: a decoder sizes its input buffer from the declaration and
    /// refuses a unit that arrives faster than it said.
    pub peak_unit_bytes: u64,
}

impl Stats {
    /// Mean bits per residual, which is what prediction is judged on.
    pub fn mean_width(&self) -> f64 {
        if self.residuals == 0 {
            0.0
        } else {
            self.residual_bits as f64 / self.residuals as f64
        }
    }

    /// The share of access units that coded one channel against another.
    pub fn matrixed_share(&self) -> f64 {
        if self.units == 0 {
            0.0
        } else {
            self.matrixed_units as f64 / self.units as f64
        }
    }

    /// The largest access unit's rate, in bits per second.
    pub fn peak_bitrate(&self, sample_rate: u32, frame_size: usize) -> u64 {
        self.peak_unit_bytes * 8 * u64::from(sample_rate) / frame_size as u64
    }

    /// The share of the stream that is not audio.
    pub fn framing_share(&self) -> f64 {
        if self.total_bytes == 0 {
            0.0
        } else {
            self.framing_bytes as f64 / self.total_bytes as f64
        }
    }
}

impl Encoder {
    pub fn new(config: Config) -> Result<Self, Unsupported> {
        let coded = config.resolve()?;
        let threads = std::sync::Arc::new(Threads::new(config.channels));
        Ok(Self {
            coded,
            sample_rate: config.sample_rate,
            channels: config.channels,
            shift: config.bits.shift_into_codec(),
            past: (0..config.channels)
                .map(|_| History::new(filter::SECOND_CONTEXT, coded.frame_size))
                .collect(),
            output_shift: vec![0; config.channels],
            second_mode: vec![None; config.channels],
            effort: Effort::default(),
            matrices: Vec::new(),
            presentations: Vec::new(),
            arrangement: Vec::new(),
            waiting: None,
            wants: Vec::new(),
            interval_shift: None,
            presentation_shift: Vec::new(),
            assignment: Vec::new(),
            permutation: (0..config.channels).collect(),
            phase_times: [std::time::Duration::ZERO; 8],
            timed: std::env::var_os("HZ_TIME").is_some(),
            threads: std::sync::Arc::clone(&threads),
            deferred: None,
            writer: Some(Writer {
                coded,
                channels: config.channels,
                hires: crate::hires::Timing::default(),
                model: crate::model::DecoderModel::default(),
                input_timing: ((coded.frame_size * (1 + INPUT_RESERVE)) as u16).wrapping_neg(),
                output_timing: 0,
                advance: (coded.frame_size * INPUT_RESERVE) as i64,
                pending_check: [0; MAX_SUBSTREAMS],
                evolution: Vec::new(),
                evolution_id: 0,
                evolution_offset: 0,
                evolution_pending: false,
                protection: None,
                since_range: [0; MAX_SUBSTREAMS],
                frames_written: 0,
                codings: vec![
                    Coding {
                        filter: crate::filter::Filter::default(),
                        codebook: crate::huffman::NONE,
                        huff_lsbs: 24,
                    };
                    config.channels
                ],
                restart_codings: vec![
                    Coding {
                        filter: crate::filter::Filter::default(),
                        codebook: crate::huffman::NONE,
                        huff_lsbs: 24,
                    };
                    config.channels
                ],
                write_fir: vec![true; config.channels],
                restart_write_fir: vec![true; config.channels],
                write_iir: vec![false; config.channels],
                restart_write_iir: vec![false; config.channels],
                write_params: vec![true; config.channels],
                restart_write_params: vec![true; config.channels],
                samples: vec![vec![0; coded.frame_size]; config.channels],
                planes: vec![vec![0; coded.frame_size]; config.channels],
                output_shift: vec![0; config.channels],
                dynamic_range: [None; MAX_SUBSTREAMS],
                since_restart: 0,
                stats: Stats {
                    // It starts full, so anything lower is something spent.
                    lowest_advance: (coded.frame_size * INPUT_RESERVE) as i64,
                    ..Stats::default()
                },
                prepared: Vec::new(),
                prepared_shifts: Vec::new(),
                prepared_checks: Vec::new(),
                interval_decided: Vec::new(),
                interval_planes: Vec::new(),
                interval_restart: Vec::new(),
                matrices: Vec::new(),
                presentations: Vec::new(),
                arrangement: Vec::new(),
                assignment: Vec::new(),
                presentation_shift: Vec::new(),
                max_shift: crate::frame::MAX_OUTPUT_SHIFT,
                max_output_bits: crate::frame::CODEC_BITS as u8,
            }),
            decided: None,
            back: Some(IntervalDecider {
                channels: config.channels,
                coded,
                // A pool of its own, of two threads for every three channels,
                // and never fewer than four or than the channels there are: a
                // stereo on one thread decided its two channels one after the
                // other. Sharing the encoder's, the blocks took threads the
                // preparing of the next interval was waiting for, and that is
                // the longer of the two: five minutes of an overlay at twelve
                // channels took 11.3 s so, and on a pool of its own 13.5 s at
                // three threads, 11.5 at four, 10.1 at six, 10.2 at eight
                // and 10.7 at sixteen. With `--fast` the preparing is lighter
                // and the blocks are the longer: 7.4 s at six, 7.2 at eight,
                // 7.5 at twelve — and at full effort eight is as good as six.
                threads: std::sync::Arc::new(Threads::exactly(
                    (2 * config.channels)
                        .div_ceil(3)
                        .max(config.channels.min(4)),
                )),
                interval_decided: vec![Vec::new(); config.channels],
                interval_planes: vec![Vec::new(); config.channels],
                interval_restart: vec![Decided::default(); config.channels],
                past_residuals: vec![Vec::new(); config.channels],
                prepared: Vec::new(),
                past: Vec::new(),
                second_mode: Vec::new(),
                stats: Stats::default(),
            }),
            fronted: None,
            intervals_decided: 0,
            evolution: Vec::new(),
            evolution_id: 0,
            evolution_offset: 0,
            evolution_pending: false,
            prepared: vec![Vec::new(); config.channels],
            prepared_shifts: Vec::new(),
            prepared_checks: Vec::new(),
            max_shift: crate::frame::MAX_OUTPUT_SHIFT,
            max_output_bits: crate::frame::CODEC_BITS as u8,
            pending: Vec::with_capacity(RESTART_INTERVAL as usize),
            spare: Vec::new(),
            carry: Vec::new(),
            carry_frames: 0,
            unit_scratch: Vec::new(),
            candidates: vec![Vec::new(); config.channels],
            dynamic_range: [None; MAX_SUBSTREAMS],
            decorrelation: true,
            since_restart: 0,
            stats: Stats {
                // It starts full, so anything lower is something spent.
                lowest_advance: (coded.frame_size * INPUT_RESERVE) as i64,
                ..Stats::default()
            },
        })
    }

    /// Frames in one access unit.
    pub fn frame_size(&self) -> usize {
        self.coded.frame_size
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Frames encoded so far.
    pub fn frames_written(&self) -> u64 {
        self.writer().frames_written
    }

    /// The writer, which is only ever out of the encoder inside
    /// [`Encoder::advance`].
    fn writer(&self) -> &Writer {
        self.writer
            .as_ref()
            .expect("the writer is in the encoder between calls")
    }

    fn writer_mut(&mut self) -> &mut Writer {
        self.writer
            .as_mut()
            .expect("the writer is in the encoder between calls")
    }

    /// What the encoder has been choosing.
    pub fn stats(&self) -> Stats {
        // What deciding counts and what writing counts are different fields
        // of the same record, kept apart while the two run side by side.
        let written = &self.writer().stats;
        Stats {
            orders: written.orders,
            iir_orders: written.iir_orders,
            filter_blocks: written.filter_blocks,
            params_restated: written.params_restated,
            codebooks: written.codebooks,
            matrixed_units: written.matrixed_units,
            units: written.units,
            residual_bits: written.residual_bits,
            residuals: written.residuals,
            framing_bytes: written.framing_bytes,
            total_bytes: written.total_bytes,
            starved: written.starved,
            lowest_advance: written.lowest_advance,
            peak_unit_bytes: written.peak_unit_bytes,
            filters_restated: self
                .back
                .as_ref()
                .map_or(0, |back| back.stats.filters_restated),
            ..self.stats.clone()
        }
    }

    /// Carry a payload in the next access unit's extra data.
    ///
    /// Written as an Evolution frame after the last substream, under the
    /// identifier given — eleven is object audio metadata — and then dropped:
    /// what rides here describes the unit it rides in, so it is stated per
    /// unit or not at all. A unit given nothing carries no extra data, which
    /// is how the format says "nothing here": there is no empty shape.
    ///
    /// Only a stream that declared Evolution extra data in its major sync may
    /// use this, which today means a sixteen-element one. A decoder reads the
    /// shape from that flag and not from the block, so writing an Evolution
    /// frame into a stream that declared an opaque payload hands it to
    /// something that will not look inside it.
    ///
    /// # Panics
    /// In debug, if the stream declared no Evolution extra data.
    pub fn set_evolution(&mut self, id: u32, payload: &[u8]) {
        self.set_evolution_at(id, payload, 0);
    }

    /// The same, taking effect `sample_offset` samples into the unit rather
    /// than at its start.
    ///
    /// A master states where its objects are at a sample, not at an access
    /// unit, and a reference stream carries that: its payloads say how far
    /// into the unit they apply, and a decoder dates what they say from
    /// there. Nought is the start of the unit, and is written as no offset at
    /// all. Anything from the unit's length up is not inside the unit.
    pub fn set_evolution_at(&mut self, id: u32, payload: &[u8], sample_offset: u32) {
        debug_assert!(
            self.coded.flags & (1 << 12) != 0,
            "this stream's major sync says its extra data is not an Evolution frame"
        );
        debug_assert!(
            (sample_offset as usize) < self.coded.frame_size,
            "a sample offset of {sample_offset} is not inside a unit of {}",
            self.coded.frame_size
        );
        self.evolution.clear();
        self.evolution.extend_from_slice(payload);
        self.evolution_id = id;
        self.evolution_offset = sample_offset;
        self.evolution_pending = true;
    }

    /// Sign each Evolution frame's protection field with this key, from the
    /// next access unit on.
    ///
    /// Without one the field holds a constant, which parses identically and
    /// verifies against nothing. What the digest covers, and why the key is
    /// not this project's to provide, is in [`crate::protection`].
    pub fn set_evolution_key(&mut self, key: &[u8]) {
        self.writer_mut().protection = Some(hz_core::hmac::HmacSha256::new(key));
    }

    /// Ask a presentation for a dynamic range gain, from the next access unit
    /// on.
    ///
    /// Indexed by substream, which is how the directory carries it: substream
    /// 0 is the two-channel presentation, the last is the widest. `None`
    /// withdraws nothing — a schedule that has begun cannot be stopped, only
    /// changed — so passing it after a gain has been set leaves the last one
    /// standing and stops refreshing it, which a decoder reports as a lapsed
    /// schedule. Set a gain of zero to mean "no reduction" instead.
    ///
    /// **What gain to ask for is not this crate's decision.** The named
    /// dynamic range characteristics are a phase 2 question and need a
    /// reference encoder running under a licence its owner issued; this writes
    /// what it is told and enforces only what the format enforces.
    pub fn set_dynamic_range(&mut self, substream: usize, gain: Option<DynamicRange>) {
        if substream < self.coded.substreams {
            self.dynamic_range[substream] = gain;
        }
    }

    /// Whether to decorrelate the early channels of a hierarchy of
    /// presentations, which is on by default and changes nothing a decoder
    /// hands back — only how much of the stream it takes. Off, the stream is
    /// what it was before the decorrelation existed, which is what measuring
    /// it needs.
    pub fn set_decorrelation(&mut self, on: bool) {
        self.decorrelation = on;
    }

    /// What a decoder stopping after `substream` hands back.
    ///
    /// The rows compute that presentation from the internal channels: one per
    /// channel it carries, reading every channel the stream has. They are not
    /// lifting steps and nothing inverts them — a full decode applies the last
    /// substream's matrices and never these — so a fold may be any set of
    /// gains the coefficient field can say, which is anything under two.
    ///
    /// Stated once and standing until stated again, like everything else a
    /// restart header carries. An empty list puts the substream back to
    /// declaring a subset of the coding matrices, which hands back the leading
    /// channels as they were written.
    ///
    /// The last substream's presentation is the programme itself and cannot be
    /// set: that one declares the matrices that make the decode lossless.
    pub fn set_presentation(&mut self, substream: usize, rows: &[Primitive]) {
        if substream + 1 >= self.coded.substreams {
            return;
        }
        if self.presentations.len() <= substream {
            self.presentations.resize(substream + 1, Vec::new());
        }
        self.presentations[substream].clear();
        self.presentations[substream].extend_from_slice(rows);
    }

    /// What turns the stored channels back into the elements.
    ///
    /// Stating it makes the stream carry a **hierarchy of presentations**
    /// rather than the elements themselves: the leading channels hold the
    /// stereo fold, the ones after them what the 5.1 adds, and so on, with the
    /// last few carrying what restores the elements. Which is the only way an
    /// early-stopping decoder can be handed a real fold — its matrices can
    /// only read the channels its own substream carries, so the fold has to
    /// already be there. [`crate::hierarchy`] builds the rows and
    /// [`crate::arrange::arrange`] the steps.
    ///
    /// Given in the order a decoder applies them, which is the order they are
    /// declared. The encoder undoes them in the opposite one, before the
    /// coding matrices and after the dead bits come off, so that a decode runs
    /// the whole thing backwards and lands exactly where it started.
    ///
    /// Takes effect at the next restart, with the presentations that go with
    /// it, and both together — see the `waiting` field. Stating it again
    /// before then replaces what was waiting.
    ///
    /// Refused, and nothing changed, if there are more steps than one
    /// substream's declaration can hold.
    pub fn set_presentations(&mut self, presentations: &[crate::hierarchy::Presentation]) {
        self.waiting = Some(presentations.to_vec());
    }

    /// How hard to search from here on. See [`Effort`].
    pub fn set_effort(&mut self, effort: Effort) {
        self.effort = effort;
    }

    /// How many channels each presentation the stream declares carries, or
    /// zero where it declares none.
    ///
    /// This is what the substream split is *for*: a decoder may stop after any
    /// of them and get a complete, correct mix. Four substreams that only
    /// decode as a whole would be four substreams for nothing, so the harness
    /// checks each one.
    pub fn presentations(&self) -> [usize; crate::format::MAX_PRESENTATIONS] {
        self.coded.presentation_channels()
    }

    /// Stream position to input channel.
    ///
    /// The format's channel order is not a WAV file's past 5.1, and the two
    /// decoders disagree about which to report: FFmpeg hands back the WAV
    /// order, `truehdd` the format's. Both are right about the bitstream and
    /// neither is a property of it, so anything comparing against a decoder's
    /// output has to know which one it is looking at.
    pub fn channel_order(&self) -> &[usize] {
        &self.coded.order[..self.channels]
    }

    /// How much of the decoder's input buffer this stream gives itself, in
    /// samples — what [`Stats::lowest_advance`] is measured against.
    pub fn input_reserve(&self) -> i64 {
        (self.coded.frame_size * INPUT_RESERVE) as i64
    }

    /// The peak bitrate the major sync declares, in bits per second.
    ///
    /// What a decoder sizes its input buffer from, and what every access unit
    /// written has to stay under — see [`Stats::peak_unit_bytes`].
    pub fn declared_peak_bitrate(&self) -> u64 {
        u64::from(self.coded.peak_bitrate) * u64::from(self.sample_rate) / 16
    }

    /// Encode one access unit from interleaved input.
    ///
    /// # Panics
    /// If `interleaved` is not exactly one unit's worth. A short unit is only
    /// legal at the end of a stream, where [`Encoder::finish`] handles it.
    /// Returns the bytes of a whole restart interval, or nothing.
    ///
    /// The encoder holds an interval before it writes one, because the
    /// decisions that stand for an interval — the matrices, the second filter
    /// each channel carries — are better made on the interval they will be
    /// applied to than on the one before it. And it works on four intervals
    /// at once — one's folds, the one before it prepared, the one before that
    /// decided block by block, the one before that written — see
    /// [`Encoder::advance`]. So all but one push in [`RESTART_INTERVAL`] hands
    /// back nothing and the last hands back the lot *of the interval three
    /// before*; the first interval's bytes come with the fourth's last push,
    /// and [`Encoder::finish`] hands back what is left. A
    /// caller appends what it is given either way, which is what every caller
    /// already did; what it reads of [`Encoder::stats`] is what has been
    /// written, which is everything once it has finished.
    pub fn push(&mut self, interleaved: &[i32]) -> Vec<u8> {
        assert_eq!(
            interleaved.len(),
            self.coded.frame_size * self.channels,
            "push takes exactly one access unit; use finish for the last one"
        );
        self.hold(interleaved, self.coded.frame_size, 0);
        if self.pending.len() as u64 == RESTART_INTERVAL {
            self.advance()
        } else {
            Vec::new()
        }
    }

    /// Encode any number of frames given as one plane per channel, laid end
    /// to end (`channels * frames` samples, channel-major), in whatever length
    /// the source produces them. Whole access units are cut from the planes
    /// and written straight into the interval being held, with no unit-sized
    /// interleaving buffer in between; what does not make a whole unit is
    /// kept for the next call, and [`Encoder::finish_carried`] ends the stream
    /// with it. Hands back bytes as [`Encoder::push`] does. Do not mix with
    /// [`Encoder::push`] on one stream.
    ///
    /// # Panics
    /// If `planes` is not `channels * frames` long.
    pub fn push_planes(&mut self, planes: &[i32], frames: usize) -> Vec<u8> {
        let channels = self.channels;
        let unit = self.coded.frame_size;
        assert_eq!(planes.len(), channels * frames, "one plane per channel");
        let mut out = Vec::new();
        let mut at = 0;
        while at < frames {
            let take = (unit - self.carry_frames).min(frames - at);
            if self.carry_frames == 0 && take == unit {
                self.hold_with(unit, 0, |samples| {
                    for (channel, plane) in planes.chunks_exact(frames).enumerate() {
                        for (frame, sample) in plane[at..at + unit].iter().enumerate() {
                            samples[frame * channels + channel] = *sample;
                        }
                    }
                });
            } else {
                self.carry.resize(unit * channels, 0);
                for (channel, plane) in planes.chunks_exact(frames).enumerate() {
                    for (frame, sample) in plane[at..at + take].iter().enumerate() {
                        self.carry[(self.carry_frames + frame) * channels + channel] = *sample;
                    }
                }
                self.carry_frames += take;
                if self.carry_frames < unit {
                    at += take;
                    continue;
                }
                let carried = std::mem::take(&mut self.carry);
                self.hold(&carried, unit, 0);
                self.carry = carried;
                self.carry_frames = 0;
            }
            at += take;
            if self.pending.len() as u64 == RESTART_INTERVAL {
                out.extend_from_slice(&self.advance());
            }
        }
        out
    }

    /// End a stream fed by [`Encoder::push_planes`]: the frames it carries
    /// become the short last unit, and everything not yet written follows.
    pub fn finish_carried(&mut self) -> Vec<u8> {
        let carried = std::mem::take(&mut self.carry);
        let frames = self.carry_frames;
        self.carry_frames = 0;
        let out = self.finish(&carried[..frames * self.channels]);
        self.carry = carried;
        out
    }

    /// Take one access unit's samples, and its payload, into the interval
    /// being held.
    fn hold(&mut self, interleaved: &[i32], frames: usize, padding: usize) {
        self.hold_with(frames, padding, |samples| {
            samples[..interleaved.len()].copy_from_slice(interleaved);
            samples[interleaved.len()..].fill(0);
        });
    }

    /// As [`Encoder::hold`], with the unit's samples written by `fill` into
    /// the buffer they will live in.
    fn hold_with(&mut self, frames: usize, padding: usize, fill: impl FnOnce(&mut [i32])) {
        let width = self.coded.frame_size * self.channels;
        let mut held = self.spare.pop().unwrap_or_default();
        held.samples.resize(width, 0);
        fill(&mut held.samples);
        held.frames = frames;
        held.padding = padding;
        held.payload.clear();
        held.evolution = self.evolution_pending.then(|| {
            held.payload.extend_from_slice(&self.evolution);
            self.evolution_id
        });
        held.evolution_offset = self.evolution_offset;
        self.evolution.clear();
        self.evolution_offset = 0;
        self.evolution_pending = false;
        self.pending.push(held);
    }

    /// Encode a final, possibly short, access unit.
    ///
    /// The unit is padded to full length and carries a marker saying how many
    /// samples to drop, which is how the format expresses a stream whose
    /// length is not a whole number of units.
    ///
    /// An empty tail writes nothing: a stream whose length is a whole number
    /// of units is already finished, and a unit that is entirely padding
    /// decodes to no samples and makes a decoder say so. What it hands back
    /// is everything not yet handed back — the interval [`Encoder::push`] was
    /// holding back, and the last one.
    ///
    /// # Panics
    /// If `interleaved` is longer than one unit.
    pub fn finish(&mut self, interleaved: &[i32]) -> Vec<u8> {
        let frames = interleaved.len() / self.channels;
        assert!(
            frames <= self.coded.frame_size,
            "finish takes at most one access unit"
        );
        if frames > 0 {
            self.hold(interleaved, frames, self.coded.frame_size - frames);
        }
        let mut out = self.advance();
        out.extend_from_slice(&self.drain());
        if self.timed {
            let names = [
                "build", "copy", "suggest", "prepare", "matrices", "second", "interval", "write",
            ];
            let total: f64 = self.phase_times.iter().map(|t| t.as_secs_f64()).sum();
            eprintln!(
                "time: {} — {:.2} s of the encoder's own",
                names
                    .iter()
                    .zip(&self.phase_times)
                    .map(|(name, t)| format!("{name} {:.2} s", t.as_secs_f64()))
                    .collect::<Vec<_>>()
                    .join(", "),
                total
            );
        }
        out
    }

    /// Add what a phase took, when timing is on.
    fn timed_phase<T>(&mut self, phase: usize, body: impl FnOnce(&mut Self) -> T) -> T {
        if !self.timed {
            return body(self);
        }
        let start = std::time::Instant::now();
        let out = body(self);
        self.phase_times[phase] += start.elapsed();
        out
    }

    /// An interval is complete: work out its folds while the one before it
    /// is prepared, the one before that is decided block by block and the
    /// one before that is written, and hand that one back.
    ///
    /// # Why three intervals late
    ///
    /// Working out an interval's folds is a search over its own samples and
    /// nothing else, and with folds it was a quarter of an encode, on one
    /// thread. Writing an interval is on one thread too, and reads nothing the
    /// deciding of the next one changes — every interval opens with a restart
    /// header, so what the decoder held before it is nothing either way. And
    /// deciding an interval's blocks keeps a thread a channel busy and no
    /// more, while preparing the next reads only the samples before it. So
    /// the four run side by side, each on its own interval: the writing with
    /// the [`Writer`] and the blocks with the [`IntervalDecider`] taken out of
    /// the encoder, the preparing with the rest. The price is that
    /// [`Encoder::push`] hands back each interval three intervals after it is
    /// complete, and [`Encoder::finish`] hands back the last four. The stream is byte for byte the same: every interval is worked
    /// out and decided from the same samples, the same request and the same
    /// decisions before it, and written with the dynamic range the caller had
    /// asked for when it was complete.
    fn advance(&mut self) -> Vec<u8> {
        let held = std::mem::take(&mut self.pending);
        if held.is_empty() {
            return Vec::new();
        }
        // What an interval's restart header describes is decided from what
        // was asked for when it is complete, and before anything reads it.
        // 🔴 The coding matrices used to be suggested on the elements and then
        // costed on the hierarchy, and found nothing: every weight was fitted
        // to a signal the channel no longer carried.
        if self.since_restart == 0
            && let Some(next) = self.waiting.take()
        {
            self.wants = next;
        }
        let planning = self.planning();
        let dynamic_range = self.dynamic_range;
        let planned = self.deferred.take();
        let fronted = self.fronted.take();
        let decided = self.decided.take();
        let mut writer = self.writer.take().expect("the writer is in the encoder");
        let mut back = self
            .back
            .take()
            .expect("the block decider is in the encoder");
        // Four intervals at once: this one's folds worked out, the one before
        // it prepared, the one before that decided block by block, and the one
        // before that written. None reads what another changes — see
        // [`Planning`], [`IntervalDecider`] and [`Writer`].
        let (planning, front, backed, written, writer, back) = std::thread::scope(|scope| {
            let plan = scope.spawn(|| planning.plan(&held));
            let write = scope.spawn(move || {
                let written = decided.map(|record| writer.write(record));
                (written, writer)
            });
            let decide = scope.spawn(move || {
                let backed = fronted.map(|front| back.decide(front));
                (backed, back)
            });
            let front = planned.map(|planned| self.decide(planned));
            let (written, writer) = write.join().expect("writing an interval does not panic");
            let (backed, back) = decide
                .join()
                .expect("deciding an interval's blocks does not panic");
            (
                plan.join().expect("working out an interval does not panic"),
                front,
                backed,
                written,
                writer,
                back,
            )
        });
        self.writer = Some(writer);
        self.back = Some(back);
        self.fronted = front;
        self.decided = self.take_what_was_decided(backed);
        self.deferred = Some(Deferred {
            held,
            planning,
            dynamic_range,
        });
        self.take_what_was_written(written)
    }

    /// The bytes of an interval just written, with its units' buffers put
    /// back rather than dropped: an interval is a megabyte or so at sixteen
    /// channels and there is one of them a third of a second.
    fn take_what_was_written(
        &mut self,
        written: Option<(Vec<u8>, Vec<Held>, std::time::Duration)>,
    ) -> Vec<u8> {
        let Some((out, held, took)) = written else {
            return Vec::new();
        };
        if self.timed {
            self.phase_times[7] += took;
        }
        self.spare.extend(held);
        out
    }

    /// An interval whose blocks were just decided, with the time it took
    /// counted.
    fn take_what_was_decided(
        &mut self,
        backed: Option<(Record, std::time::Duration)>,
    ) -> Option<Record> {
        let (record, took) = backed?;
        if self.timed {
            self.phase_times[6] += took;
        }
        Some(record)
    }

    /// Decide an interval's blocks here and now, rather than beside anything.
    fn decide_blocks(&mut self, front: Front) -> Record {
        let backed = self
            .back
            .as_mut()
            .expect("the block decider is in the encoder")
            .decide(front);
        self.take_what_was_decided(Some(backed))
            .expect("an interval was just decided")
    }

    /// Everything not yet written, written, in order: the interval decided,
    /// then the one prepared, then the one worked out.
    fn drain(&mut self) -> Vec<u8> {
        let mut out = Vec::new();
        if let Some(record) = self.decided.take() {
            let written = self.writer_mut().write(record);
            out.extend_from_slice(&self.take_what_was_written(Some(written)));
        }
        if let Some(front) = self.fronted.take() {
            let record = self.decide_blocks(front);
            let written = self.writer_mut().write(record);
            out.extend_from_slice(&self.take_what_was_written(Some(written)));
        }
        if let Some(planned) = self.deferred.take() {
            let front = self.decide(planned);
            let record = self.decide_blocks(front);
            let written = self.writer_mut().write(record);
            out.extend_from_slice(&self.take_what_was_written(Some(written)));
        }
        out
    }

    /// A [`Planning`] for the next interval, with what it reads.
    fn planning(&self) -> Planning {
        Planning {
            wants: self.wants.clone(),
            channels: self.channels,
            coded: FrameSize {
                frame_size: self.coded.frame_size,
            },
            decorrelation: self.decorrelation,
            threads: std::sync::Arc::clone(&self.threads),
            interval_shift: None,
            arrangement: Vec::new(),
            presentations: Vec::new(),
            assignment: Vec::new(),
            presentation_shift: Vec::new(),
            permutation: Vec::new(),
            stats: Stats::default(),
            took: std::time::Duration::ZERO,
        }
    }

    /// Take what was decided for an interval about to be written.
    fn take_the_plan(&mut self, planning: Planning) {
        self.interval_shift = planning.interval_shift;
        self.arrangement = planning.arrangement;
        self.presentations = planning.presentations;
        self.assignment = planning.assignment;
        self.presentation_shift = planning.presentation_shift;
        self.permutation = planning.permutation;
        self.stats.folds_asked += planning.stats.folds_asked;
        self.stats.folds_carried += planning.stats.folds_carried;
        self.stats.folds_over_the_domain += planning.stats.folds_over_the_domain;
        if self.timed {
            self.phase_times[0] += planning.took;
        }
    }

    /// Decide an interval that is complete and worked out, and hand over
    /// what the writing needs.
    fn decide(&mut self, deferred: Deferred) -> Front {
        let Deferred {
            held,
            planning,
            dynamic_range,
        } = deferred;
        if self.since_restart == 0 {
            self.take_the_plan(planning);
        }
        self.timed_phase(3, |encoder| encoder.prepare(&held));
        if self.effort != Effort::Realtime {
            self.timed_phase(5, |encoder| encoder.decide_second_filters(held.len()));
        }
        self.intervals_decided += 1;
        let channels = self.channels;
        let width = self.coded.frame_size;
        // What the blocks of this interval are decided against is the past
        // before it; what the next interval is prepared against is the past
        // after it — and the past is the prepared samples and nothing else,
        // so it moves on here rather than once the blocks are decided.
        let before = self.past.clone();
        for (history, plane) in self.past.iter_mut().zip(&self.prepared) {
            for unit in plane.chunks_exact(width).take(held.len()) {
                history.push(unit);
            }
        }
        let second_mode = self.second_mode.clone();
        // The buffers the preparing filled go with the interval; the next one
        // starts from empty ones of the same shape, which it sizes itself.
        let record = Record {
            held,
            dynamic_range,
            prepared: std::mem::replace(&mut self.prepared, vec![Vec::new(); channels]),
            prepared_shifts: std::mem::take(&mut self.prepared_shifts),
            prepared_checks: std::mem::take(&mut self.prepared_checks),
            interval_decided: Vec::new(),
            interval_planes: Vec::new(),
            interval_restart: Vec::new(),
            matrices: self.matrices.clone(),
            presentations: self.presentations.clone(),
            arrangement: self.arrangement.clone(),
            assignment: self.assignment.clone(),
            presentation_shift: self.presentation_shift.clone(),
            max_shift: self.max_shift,
            max_output_bits: self.max_output_bits,
        };
        Front {
            record,
            past: before,
            second_mode,
            order: self.effort.fixed_order(),
        }
    }

    /// Which channels each destination might be coded against, over the
    /// interval it is about to be coded in.
    ///
    /// This used to be decided on the thousand samples immediately *before*
    /// the interval, because the encoder wrote each unit as it arrived and had
    /// nothing else. A correlation is a property of the passage it is measured
    /// over, and the passage that matters is the one the matrix will stand
    /// for.
    ///
    /// Only the sources and the weights are decided here. Whether to use them
    /// at all is a question about coded size, and that is answered by the unit
    /// that opens the interval — see [`Encoder::choose_matrix`].
    fn suggest_matrices(&mut self, units: usize) {
        // Only the last substream of an object programme may state a step, and
        // only for a destination no earlier substream describes: a matrix
        // stated twice has to be stated the same way both times.
        let last = self.coded.ranges[self.coded.substreams - 1];
        let exclusive = if last.sync == crate::format::RestartSync::C && self.coded.substreams >= 2
        {
            Some(self.coded.ranges[self.coded.substreams - 2].last + 1)
        } else {
            None
        };
        // What each destination needs to know, gathered first: the substream
        // it belongs to bounds which channels may be its sources.
        let mut bounds = vec![(0usize, false); self.channels];
        for index in 0..self.coded.substreams {
            let range = self.coded.ranges[index];
            for (dest, slot) in bounds
                .iter_mut()
                .enumerate()
                .take(range.last + 1)
                .skip(range.first)
            {
                *slot = (range.last, exclusive.is_some_and(|first| dest >= first));
            }
        }

        // One destination's candidates are read from the whole interval and
        // written nowhere else, so the destinations are independent — and this
        // was the largest thing the encoder still did on one thread.
        let interval = &self.prepared;
        let frames = self.coded.frame_size;
        let body = |dest: usize, into: &mut Vec<Candidate>| {
            let (available, aimable) = bounds[dest];
            into.clear();
            let sources = Self::best_partners(interval, dest, available);
            let target = &interval[dest];
            for count in 1..=sources.len() {
                let windows: Vec<&[i32]> = sources[..count]
                    .iter()
                    .map(|source| interval[*source].as_slice())
                    .collect();
                // One source keeps its own exact estimator: a single
                // least-squares weight is a ratio of two integer sums and
                // needs no floating point to be right.
                let weights = if count == 1 {
                    matrix::best_weight(target, windows[0], matrix::FRACTION).map(|w| vec![w])
                } else {
                    matrix::best_weights(target, &windows, matrix::FRACTION)
                };
                let Some(weights) = weights else { continue };
                let (weights, steps) = if aimable {
                    Self::line_through(frames, target, &windows, &weights, units)
                } else {
                    let steps = vec![0i32; weights.len()];
                    (weights, steps)
                };
                into.push(Candidate {
                    sources: sources[..count].to_vec(),
                    weights,
                    steps,
                });
            }
        };

        #[cfg(feature = "parallel")]
        if let Some(pool) = &self.threads.0 {
            use rayon::prelude::*;
            pool.install(|| {
                self.candidates
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(dest, into)| body(dest, into));
            });
            return;
        }
        for (dest, into) in self.candidates.iter_mut().enumerate() {
            body(dest, into);
        }
    }

    /// A straight line through the weight over the interval.
    ///
    /// The weight is fitted on each of [`MATRIX_SEGMENTS`] slices of the
    /// interval and a least-squares line is put through those, which gives the
    /// weight at unit zero and how far it moves per unit. A whole-interval fit
    /// gives the *average* weight, and for a matrix that ramps the average is
    /// half the drift away from where it should start — which on material
    /// whose correlation moves costs more than the matrix saves.
    ///
    /// Falls back to the whole-interval weight, standing still, when a segment
    /// cannot be fitted or the line says the weight is not moving.
    fn line_through(
        frames: usize,
        target: &[i32],
        windows: &[&[i32]],
        whole: &[i32],
        units: usize,
    ) -> (Vec<i32>, Vec<i32>) {
        let still = vec![0i32; whole.len()];
        if units < MATRIX_SEGMENTS * 2 || target.len() < MATRIX_SEGMENTS * frames {
            return (whole.to_vec(), still);
        }
        let per_segment = units / MATRIX_SEGMENTS;
        let mut fitted: Vec<Vec<i32>> = Vec::with_capacity(MATRIX_SEGMENTS);
        for segment in 0..MATRIX_SEGMENTS {
            let from = segment * per_segment * frames;
            let to = from + per_segment * frames;
            let slices: Vec<&[i32]> = windows.iter().map(|w| &w[from..to]).collect();
            let piece = if slices.len() == 1 {
                matrix::best_weight(&target[from..to], slices[0], matrix::FRACTION).map(|w| vec![w])
            } else {
                matrix::best_weights(&target[from..to], &slices, matrix::FRACTION)
            };
            match piece {
                Some(piece) => fitted.push(piece),
                None => return (whole.to_vec(), still),
            }
        }

        // Least squares through the segment midpoints, in units.
        let midpoints: Vec<f64> = (0..MATRIX_SEGMENTS)
            .map(|segment| (segment * per_segment + per_segment / 2) as f64)
            .collect();
        let mean_x = midpoints.iter().sum::<f64>() / MATRIX_SEGMENTS as f64;
        let spread: f64 = midpoints.iter().map(|x| (x - mean_x) * (x - mean_x)).sum();
        if spread <= 0.0 {
            return (whole.to_vec(), still);
        }

        let mut weights = Vec::with_capacity(whole.len());
        let mut steps = Vec::with_capacity(whole.len());
        let mut moves = false;
        for source in 0..whole.len() {
            let mean_y =
                fitted.iter().map(|w| f64::from(w[source])).sum::<f64>() / MATRIX_SEGMENTS as f64;
            let covariance: f64 = midpoints
                .iter()
                .zip(&fitted)
                .map(|(x, w)| (x - mean_x) * (f64::from(w[source]) - mean_y))
                .sum();
            let slope = covariance / spread;
            let intercept = mean_y - slope * mean_x;
            let step = slope.round();
            if !intercept.is_finite() || !step.is_finite() {
                return (whole.to_vec(), still);
            }
            // The coefficient has to stay inside the field the whole interval,
            // not only at its start: the decoder accumulates the step at the
            // end of every unit.
            let end = intercept + step * units as f64;
            let limit = f64::from(1i32 << (matrix::FRACTION + 1));
            if intercept.abs() >= limit || end.abs() >= limit {
                return (whole.to_vec(), still);
            }
            moves |= step != 0.0;
            weights.push(intercept.round() as i32);
            steps.push(step as i32);
        }
        if moves {
            (weights, steps)
        } else {
            (whole.to_vec(), still)
        }
    }

    /// Prepare every unit of the held interval: de-interleave it through the
    /// permutation, take its dead bits off, take the cascade off, certify what
    /// each substream puts out, decide the matrices on the whole of it, and
    /// apply them.
    ///
    /// The units are independent of one another once each channel's dead-bit
    /// shift is known, so they go across the pool in two passes, one before
    /// the matrices are decided and one after; `prepared` holds the channels
    /// as they stand before the matrices between the two, which is what the
    /// matrices are suggested and judged on. What is not independent is the
    /// shift a silent unit inherits from the unit before it, and that is a
    /// pass of its own between them which takes no time at all. This used to
    /// walk the units one after another on one thread — two and a half
    /// seconds of twenty-five on four minutes of a programme — and gather the
    /// interval for the matrices a second time, with the cascade taken off
    /// twice.
    fn prepare(&mut self, held: &[Held]) {
        let width = self.coded.frame_size;
        let channels = self.channels;
        let units = held.len();
        for plane in &mut self.prepared {
            plane.clear();
            plane.resize(units * width, 0);
        }
        self.prepared_shifts.clear();
        self.prepared_shifts.resize(units * channels, 0);
        self.prepared_checks.clear();
        self.prepared_checks.resize(units, [0; MAX_SUBSTREAMS]);

        // Unit-major scratch: each unit's channels side by side, so that a
        // unit is one chunk a thread can own. Zeroed, so a short unit's tail
        // is the silence the format says it carries.
        let stride = channels * width;
        let mut scratch = std::mem::take(&mut self.unit_scratch);
        scratch.clear();
        scratch.resize(units * stride, 0);

        // Pass one: de-interleave, certify, and measure each channel's dead
        // bits.
        let measured: Vec<Measured> = {
            let body = |(chunk, unit): (&mut [i32], &Held)| self.unit_in(unit, chunk);
            #[cfg(feature = "parallel")]
            if let Some(pool) = &self.threads.0 {
                use rayon::prelude::*;
                pool.install(|| {
                    scratch
                        .par_chunks_mut(stride)
                        .zip(held.par_iter())
                        .map(body)
                        .collect()
                })
            } else {
                scratch
                    .chunks_mut(stride)
                    .zip(held.iter())
                    .map(body)
                    .collect()
            }
            #[cfg(not(feature = "parallel"))]
            scratch
                .chunks_mut(stride)
                .zip(held.iter())
                .map(body)
                .collect()
        };

        // The shift a silent unit holds is the one in force before it, which
        // is the one thing about a unit that depends on the unit before.
        let mut previous: Vec<u8> = self.output_shift[..channels].to_vec();
        for (index, measured) in measured.iter().enumerate() {
            for (channel, held) in previous.iter_mut().enumerate() {
                *held = measured.wasted[channel].unwrap_or(*held);
                self.prepared_shifts[index * channels + channel] = *held;
            }
            self.prepared_checks[index] = measured.check;
        }
        self.output_shift[..channels].copy_from_slice(&previous);

        // Pass two: the dead bits off, the cascade off, and what an early
        // substream then puts out certified.
        let certified: Vec<[u32; MAX_SUBSTREAMS]> = {
            let shifts = &self.prepared_shifts;
            let checks = &self.prepared_checks;
            let body = |(index, chunk): (usize, &mut [i32])| {
                self.unit_stored(
                    chunk,
                    &shifts[index * channels..(index + 1) * channels],
                    held[index].frames,
                    checks[index],
                )
            };
            #[cfg(feature = "parallel")]
            if let Some(pool) = &self.threads.0 {
                use rayon::prelude::*;
                pool.install(|| {
                    scratch
                        .par_chunks_mut(stride)
                        .enumerate()
                        .map(body)
                        .collect()
                })
            } else {
                scratch.chunks_mut(stride).enumerate().map(body).collect()
            }
            #[cfg(not(feature = "parallel"))]
            scratch.chunks_mut(stride).enumerate().map(body).collect()
        };
        self.prepared_checks.copy_from_slice(&certified);
        self.timed_phase(1, |encoder| encoder.gather(&scratch, stride));

        // The matrices, decided on the interval as it stands before them:
        // permuted, the dead bits off, the cascade off — the domain a decoder
        // applies them in, since it puts the dead bits back after them.
        // Not at real-time effort: a coding matrix is a search over every
        // channel pair of the interval, and the stream is valid without one.
        if self.effort != Effort::Realtime {
            self.timed_phase(2, |encoder| encoder.suggest_matrices(units));
            if self.since_restart == 0 {
                self.timed_phase(4, |encoder| encoder.decide_matrices());
            }
        }

        // Pass three: the matrices on, each unit at its own place on their
        // ramp.
        if !self.matrices.is_empty() {
            loop {
                self.apply_the_matrices(&mut scratch, stride);
                // 🔴 And what they left has to fit the codec's domain. A
                // decoder reconstructs a stored sample in twenty-four bits and
                // refuses the stream past them, and a matrix is a
                // least-squares fit over the interval: it takes energy off on
                // average and not on every sample. `judge_matrix` refuses a
                // candidate on the weights it was judged with; this is the
                // same bound on what is really stored, steps and all, and it
                // has the last word. A matrix that fails it is dropped and
                // the interval matrixed again with what is left, since a
                // matrix reads the channels the dropped one wrote.
                let over = self.over_the_domain(&scratch, stride);
                let before = self.matrices.len();
                let stats = &mut self.stats;
                self.matrices.retain(|matrix| {
                    let keep = !over[matrix.dest];
                    if !keep && matrix.delta.iter().any(|step| *step != 0) {
                        stats.aimed_matrices -= 1;
                    }
                    keep
                });
                let dropped = before - self.matrices.len();
                if dropped == 0 {
                    break;
                }
                if std::env::var_os("HZ_DOMAIN").is_some() {
                    eprintln!("domain: dropped {dropped} decided matrices, over {over:?}");
                }
                self.stats.matrices_refused += dropped as u64;
                self.scatter(&mut scratch, stride);
            }
            self.timed_phase(1, |encoder| encoder.gather(&scratch, stride));
        }
        self.unit_scratch = scratch;

        // What the restart header opening this interval may declare, now that
        // the interval exists: the largest shift any of its units takes, and
        // the width of the widest sample a decoder will emit — which is the
        // widest that went in, since that is what it emits.
        // 🔴 And what the early substreams state as their own output shifts,
        // which a decoder checks against the same bound: a presentation
        // written at five with the header saying four is a stream it refuses.
        let stated = self
            .presentation_shift
            .iter()
            .flatten()
            .copied()
            .max()
            .unwrap_or(0);
        self.max_shift = self
            .prepared_shifts
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .max(stated);
        // Of what went in, not of the channels as they are stored: those have
        // their dead bits off and the cascade on, and a decoder's full output
        // is the elements at their own width.
        let widest = held
            .iter()
            .flat_map(|unit| unit.samples[..unit.frames * self.channels].iter())
            .fold(0u32, |m, sample| {
                m.max((sample << self.shift).unsigned_abs())
            });
        self.max_output_bits =
            (33 - widest.leading_zeros()).clamp(1, crate::frame::CODEC_BITS) as u8;
    }

    /// The unit-major scratch, channel by channel into `prepared`.
    fn gather(&mut self, scratch: &[i32], stride: usize) {
        let width = self.coded.frame_size;
        for (index, chunk) in scratch.chunks(stride).enumerate() {
            for (channel, plane) in self.prepared.iter_mut().enumerate() {
                plane[index * width..(index + 1) * width]
                    .copy_from_slice(&chunk[channel * width..(channel + 1) * width]);
            }
        }
    }

    /// And back: `prepared` into the unit-major scratch, which is how the
    /// interval is returned to the channels as they stood before the
    /// matrices.
    fn scatter(&self, scratch: &mut [i32], stride: usize) {
        let width = self.coded.frame_size;
        for (index, chunk) in scratch.chunks_mut(stride).enumerate() {
            for (channel, plane) in self.prepared.iter().enumerate() {
                chunk[channel * width..(channel + 1) * width]
                    .copy_from_slice(&plane[index * width..(index + 1) * width]);
            }
        }
    }

    /// The decided matrices taken off the unit-major scratch, each unit at
    /// its own place on their ramp.
    fn apply_the_matrices(&self, scratch: &mut [i32], stride: usize) {
        let width = self.coded.frame_size;
        let matrices = &self.matrices;
        let body = |(index, chunk): (usize, &mut [i32])| {
            let mut planes: Vec<&mut [i32]> = chunk.chunks_mut(width).collect();
            matrix::apply_inverse(matrices, &mut planes, width, index);
        };
        #[cfg(feature = "parallel")]
        if let Some(pool) = &self.threads.0 {
            use rayon::prelude::*;
            pool.install(|| {
                scratch.par_chunks_mut(stride).enumerate().for_each(body);
            });
        } else {
            scratch.chunks_mut(stride).enumerate().for_each(body);
        }
        #[cfg(not(feature = "parallel"))]
        scratch.chunks_mut(stride).enumerate().for_each(body);
    }

    /// Which channels of the unit-major scratch hold a sample outside the
    /// domain a decoder reconstructs them in.
    fn over_the_domain(&self, scratch: &[i32], stride: usize) -> Vec<bool> {
        let width = self.coded.frame_size;
        let domains: Vec<i32> = (0..self.channels)
            .map(|channel| self.domain_of(channel))
            .collect();
        let mut over = vec![false; self.channels];
        for chunk in scratch.chunks(stride) {
            for (channel, plane) in chunk.chunks(width).enumerate().take(self.channels) {
                let domain = domains[channel];
                if !over[channel]
                    && plane
                        .iter()
                        .any(|sample| !(-domain..domain).contains(sample))
                {
                    over[channel] = true;
                }
            }
        }
        over
    }

    /// The domain a stored sample of `channel` is reconstructed in.
    ///
    /// The codec's twenty-four bits — or the whole word under restart sync
    /// word C, whose filters run in thirty-two, which is how the reference
    /// decoder holds them and why a channel of the immersive substream may
    /// store past twenty-four bits and decode.
    fn domain_of(&self, channel: usize) -> i32 {
        let wide = self.coded.ranges[..self.coded.substreams]
            .iter()
            .any(|range| {
                range.first <= channel
                    && channel <= range.last
                    && matches!(range.sync, crate::format::RestartSync::C)
            });
        if wide { i32::MAX } else { DOMAIN }
    }

    /// One unit into the codec's domain: de-interleaved through the
    /// permutation, the check the *next* restart header will carry folded
    /// per substream — each one certifies every channel up to its own last —
    /// and each channel's dead bits measured.
    fn unit_in(&self, unit: &Held, chunk: &mut [i32]) -> Measured {
        let width = self.coded.frame_size;
        let channels = self.channels;
        let mut check = [0u32; MAX_SUBSTREAMS];
        for frame in 0..unit.frames {
            for channel in 0..channels {
                let element = self.permutation[channel];
                let sample =
                    unit.samples[frame * channels + self.coded.order[element]] << self.shift;
                chunk[channel * width + frame] = sample;
                // The shift wraps at eight channels. Below that it is just
                // the channel index, which is why it went unnoticed until
                // there were sixteen of them — where shifting a 24-bit value
                // by fifteen leaves the accumulator.
                let contribution = ((sample as u32) & 0x00ff_ffff) << (channel & 7);
                for (index, slot) in check.iter_mut().enumerate().take(self.coded.substreams) {
                    if channel <= self.coded.ranges[index].last {
                        *slot ^= contribution;
                    }
                }
            }
        }
        // 🔴 Held, not measured, while presentations are stated. A restart
        // header states the matrices once for a whole interval and they are
        // written in the units the shifts make, so a unit that shifted
        // differently would be decoded by matrices meant for another. Those
        // shifts are settled before any of it is prepared — see
        // [`Encoder::settle_the_shifts`]. Otherwise a silent unit measures
        // nothing, and takes what was in force before it.
        let mut wasted = [None; crate::format::MAX_CHANNELS];
        for (channel, slot) in wasted.iter_mut().enumerate().take(channels) {
            *slot = match &self.interval_shift {
                Some(held) => held.get(channel).copied(),
                None => wasted_bits(&chunk[channel * width..(channel + 1) * width]),
            };
        }
        Measured { check, wasted }
    }

    /// One unit as it will be coded: its dead bits off, the cascade off —
    /// first, because a decoder puts it back last, so what is coded from here
    /// on is the hierarchy of presentations rather than the elements — and
    /// then what each early substream certifies.
    ///
    /// 🔴 What an early substream certifies is what its decoder puts out, and
    /// with a presentation stated that is the fold — its rows run over these
    /// very channels — and not the leading elements the check was folded
    /// over on the way in. A stream certifying the elements there decodes
    /// with every checksum right and its lossless check wrong at every
    /// restart, which the reference decoder warns about and goes on, and
    /// which the one in `tests/decoder` refuses.
    fn unit_stored(
        &self,
        chunk: &mut [i32],
        shifts: &[u8],
        frames: usize,
        mut check: [u32; MAX_SUBSTREAMS],
    ) -> [u32; MAX_SUBSTREAMS] {
        let width = self.coded.frame_size;
        for (channel, shift) in shifts.iter().enumerate() {
            if *shift > 0 {
                for sample in &mut chunk[channel * width..(channel + 1) * width] {
                    *sample >>= shift;
                }
            }
        }
        let mut planes: Vec<&mut [i32]> = chunk.chunks_mut(width).collect();
        for step in self.arrangement.iter().rev() {
            crate::arrange::unapply_within(step, &mut planes, width);
        }
        let early = self.coded.substreams.saturating_sub(1);
        for (substream, slot) in check.iter_mut().enumerate().take(early) {
            if let Some(rows) = self.presentations.get(substream)
                && !rows.is_empty()
            {
                *slot = presentation_check(
                    &planes,
                    self.coded.ranges[substream].max_matrix,
                    rows,
                    self.presentation_shift.get(substream),
                    frames,
                );
            }
        }
        check
    }

    /// Which second filter each channel carries through this interval.
    ///
    /// Decided on the interval it will be applied to, prepared exactly as it
    /// will be coded — matrices on, dead bits off. It used to be decided on
    /// the forty-odd blocks immediately *before*, because the encoder wrote
    /// each unit as it arrived: a second filter is a fixed design that stands
    /// for as long as the signal keeps its roll-off, and the roll-off that
    /// matters is the one it is about to meet.
    ///
    /// The judge is still the block search with its exact costs, run in every
    /// mode — no model of the residuals' width will do, and that was measured.
    fn decide_second_filters(&mut self, units: usize) {
        if self.since_restart != 0 {
            return;
        }
        let width = self.coded.frame_size;
        let block = crate::format::MAX_BLOCK.min(40);
        let ahead = (units * width).min(filter::TRIAL_BLOCKS * block);
        let restart = self.intervals_decided;
        let effort = self.effort;

        let past = &self.past;
        let prepared = &self.prepared;
        let threads = &self.threads;
        let modes = &mut self.second_mode;

        // One channel's decision reads that channel and nothing else, and it
        // is the most expensive thing the encoder does at a restart — it codes
        // forty-odd blocks in every mode. It runs across the channels for the
        // same reason the block search does.
        let body = |channel: usize, mode: &mut Option<usize>| {
            // The fit needs a context before the first block it judges, and
            // that is the previous interval's tail — the only thing about this
            // decision that still looks backwards, and the same context every
            // filter choice in the interval will start from.
            let behind = past[channel].window();
            let carried = behind.len().min(filter::FIT_CONTEXT);
            let mut trial = Vec::with_capacity(carried + ahead);
            trial.extend_from_slice(&behind[behind.len() - carried..]);
            trial.extend_from_slice(&prepared[channel][..ahead]);
            *mode = filter::pick_second(&trial, *mode, effort, restart);
        };

        #[cfg(feature = "parallel")]
        if let Some(pool) = &threads.0 {
            use rayon::prelude::*;
            pool.install(|| {
                modes
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(channel, mode)| body(channel, mode));
            });
            return;
        }
        let _ = threads;
        modes
            .iter_mut()
            .enumerate()
            .for_each(|(channel, mode)| body(channel, mode));
    }

    /// Every early presentation's rows over what the channels hold, each at
    /// whichever output shift takes the fewest; `None` if any cannot be
    /// written.
    #[allow(clippy::too_many_arguments)]
    fn presentation_rows(
        threads: &Threads,
        held: &[Vec<f64>],
        internal: &[crate::hierarchy::Presentation],
        elements: usize,
        fold_log: bool,
        strict: bool,
        peaks: &[Vec<f64>],
    ) -> Option<PresentationRows> {
        let mut rows = Vec::with_capacity(internal.len() - 1);
        let mut assignment = Vec::with_capacity(internal.len() - 1);
        let mut stated: Vec<Vec<u8>> = Vec::with_capacity(internal.len() - 1);
        for (which, presentation) in internal[..internal.len() - 1].iter().enumerate() {
            // 🔴 A presentation's own shifts are **free**, and this is what the
            // reference's restating of them per substream is for. The encoder's
            // own shifting is the elements' — that is what makes the full
            // decode lossless — but what an early substream states is only how
            // loud its own output comes out, so it can be whatever makes the
            // rows writable.
            //
            // Without that, a presentation's channel `j` is written in the
            // units of *element* `j`: an unrounded low frequency bed at zero
            // among elements at four leaves a row asking for sixteen times what
            // it wants. Measured at 136.
            // Every shift the presentation could be written at is tried and
            // the one taking the fewest rows is kept: a row past what the
            // field can say costs a scaling row on top, and the rows share a
            // four-bit count with the coding matrices, which on the 7.1's
            // substream were found stopped at fifteen. Ties go to the lower
            // shift.
            //
            // And a shift per channel, each the smallest its own row needs,
            // which is kept where it takes no more rows — see
            // [`crate::hierarchy::rows_over_choosing_shifts`]. The two are
            // weighed on the rows first and then on the largest shift, the
            // precision the presentation gives up.
            // Every candidate is written and judged on its own, so they are
            // written and judged side by side, and then chosen from in the
            // order they were always tried in: the per-channel shifts first,
            // then one shift for every channel from nought up.
            //
            // 🔴 And only as precise as a fold has to be: a coefficient's
            // rounding comes out scaled up by its channel's shift, so a large
            // shift on a row that also reads a channel at full size lands the
            // presentation tens of decibels off — measured at −66 dBFS on a
            // film's 5.1 before this. Candidates past
            // [`PRESENTATION_PRECISION`] give way to those within it; where
            // none is, the rows are what one shift for every channel wrote
            // before, unless `strict`, which is how a decorrelating step asks
            // whether it can be afforded.
            //
            // 🔴 And the shift has to leave the fold room at its loudest. A
            // channel's output is its row's result shifted up, so it is
            // rounded to `2^shift`, and a fold within that of full scale is
            // rounded past it and wraps — measured as one interval of a
            // film's 5.1 coming back sign-flipped at full scale, from a
            // shift of seven on a fold that peaked there.
            let peaks = peaks.get(which);
            let fits = |order: &[u8], shifts: &[u8]| {
                let Some(peaks) = peaks else {
                    return true;
                };
                shifts.iter().enumerate().all(|(channel, shift)| {
                    let output = order.get(channel).map_or(channel, |o| usize::from(*o));
                    let peak = peaks.get(output).copied().unwrap_or(0.0);
                    peak + f64::from(*shift + 1).exp2() + FOLD_MARGIN < FULL_SCALE
                })
            };
            let tries: Vec<Option<u8>> = std::iter::once(None)
                .chain((0..=crate::frame::MAX_OUTPUT_SHIFT).map(Some))
                .collect();
            let written = threads.map(&tries, |shift| {
                let candidate = match shift {
                    None => crate::hierarchy::rows_over_choosing_shifts(
                        held,
                        presentation,
                        crate::hierarchy::PRESENTATION_BITS,
                    )
                    .map(|(rows, order, mut shifts)| {
                        shifts.resize(presentation.channels, 0);
                        (rows, order, shifts)
                    }),
                    Some(state) => crate::hierarchy::rows_over(
                        held,
                        presentation,
                        &vec![*state; elements],
                        crate::hierarchy::PRESENTATION_BITS,
                    )
                    .map(|(rows, order)| (rows, order, vec![*state; presentation.channels])),
                }?;
                let (rows, order, shifts) = &candidate;
                let precise = fits(order, shifts)
                    && presentation_error(held, presentation, rows, order, shifts)
                        <= PRESENTATION_PRECISION;
                Some((candidate, precise))
            });
            let mut candidates = Vec::with_capacity(written.len());
            let mut precise = Vec::new();
            for (candidate, is_precise) in written.into_iter().flatten() {
                if is_precise {
                    precise.push(candidate.clone());
                }
                candidates.push(candidate);
            }
            let pool = if !precise.is_empty() {
                precise
            } else if strict {
                Vec::new()
            } else {
                candidates
                    .into_iter()
                    .filter(|(_, _, shifts)| shifts.windows(2).all(|pair| pair[0] == pair[1]))
                    .collect()
            };
            let Some((row, order)) = pool
                .into_iter()
                .min_by_key(|(rows, _, shifts)| {
                    (rows.len(), shifts.iter().copied().max().unwrap_or(0))
                })
                .map(|(row, order, shifts)| {
                    stated.push(shifts);
                    (row, order)
                })
            else {
                if fold_log {
                    eprintln!(
                        "fold: the {}-channel presentation cannot be written over what the channels hold",
                        presentation.channels
                    );
                }
                return None;
            };
            rows.push(row);
            assignment.push(order);
        }
        Some((rows, assignment, stated))
    }

    /// Choose the matrices for the interval this unit opens.
    ///
    /// Decided once per restart interval and held, not chosen per access unit.
    /// Two reasons, and the second is the one that matters: a matrix costs
    /// forty-odd bits to describe, which is worth amortising over the interval
    /// rather than paying every time; and switching one turns the signal a
    /// channel carries into a different signal, which the prediction filter's
    /// carried history and fit context are then wrong about for the next
    /// quarter of a second.
    fn decide_matrices(&mut self) {
        self.matrices.clear();
        // Every destination judged at once — each against the interval and
        // the first unit as they stand, which no other destination's choice
        // changes — and then admitted in order under each substream's
        // budget. Judging them one after another was three seconds of
        // twenty-five on one thread, for four minutes of a programme.
        let judged = self.judge_matrices();
        self.stats.matrices_refused += judged.iter().map(|judged| judged.refused).sum::<u64>();
        for index in 0..self.coded.substreams {
            let range = self.coded.ranges[index];
            for (dest, judged) in judged.iter().enumerate().take(range.last + 1) {
                let Some(candidate) = &judged.best else {
                    continue;
                };
                if dest < range.first || self.matrices.iter().any(|held| held.dest == dest) {
                    continue;
                }
                if !self.has_room_for(dest) {
                    continue;
                }
                self.matrices.push(*candidate);
            }
        }
        self.aim_the_matrices();
    }

    /// The best candidate for every destination, or none where coding the
    /// channel alone is cheaper, judged across the pool — and how many each
    /// destination refused for the codec's domain.
    fn judge_matrices(&self) -> Vec<Judged> {
        let judge = |dest: usize| -> Judged {
            if self.candidates[dest].is_empty() {
                return Judged::default();
            }
            let mut scratch = vec![0i32; self.coded.frame_size];
            self.judge_matrix(dest, &mut scratch)
        };
        #[cfg(feature = "parallel")]
        if let Some(pool) = &self.threads.0 {
            use rayon::prelude::*;
            return pool.install(|| (0..self.channels).into_par_iter().map(judge).collect());
        }
        (0..self.channels).map(judge).collect()
    }
}

impl Encoder {
    /// Whether every substream that would have to declare a matrix writing
    /// `dest` still has a slot for it.
    ///
    /// A substream's count is a four-bit field, and it is shared: the coding
    /// matrices, a presentation's rows and the cascade all go through it. The
    /// cascade took fifteen of the last substream's sixteen while it was
    /// written as clamped steps, and the coding matrices — which are what make
    /// a dense channel cheap — were left with one.
    fn has_room_for(&self, dest: usize) -> bool {
        (0..self.coded.substreams).all(|index| {
            let range = self.coded.ranges[index];
            if range.last < dest {
                return true;
            }
            let last = index + 1 == self.coded.substreams;
            let coding = self
                .matrices
                .iter()
                .filter(|matrix| last || matrix.dest <= range.last)
                .count();
            let (fixed, most) = if last {
                (
                    self.arrangement.len(),
                    crate::matrix::MAX_MATRICES_IMMERSIVE,
                )
            } else {
                (
                    self.presentations.get(index).map_or(0, Vec::len),
                    crate::matrix::MAX_MATRICES,
                )
            };
            coding + fixed < most
        })
    }

    /// The channels `dest` is most nearly a weighted sum of, in the order they
    /// were picked, over the interval being held — strided to
    /// [`RANK_SAMPLES`], since a correlation estimated on a couple of thousand
    /// samples of a passage is the correlation.
    ///
    /// Greedy and by *partial* correlation: the first source is whichever
    /// explains most of the destination's energy, and each one after it is
    /// judged on what is left once the sources already chosen have been fitted
    /// and subtracted. Judging every source against the destination itself
    /// picks three channels that all carry the same thing, and the second and
    /// third then buy nothing.
    ///
    /// Restricted to what a decoder reconstructing `dest` already holds: a
    /// matrix is undone by every substream that codes its destination, and the
    /// first of those is the one `dest` is in, so a source cannot be past that
    /// substream's last channel.
    fn best_partners(interval: &[Vec<i32>], dest: usize, available: usize) -> Vec<usize> {
        // Every source of every destination is scored against the whole
        // interval, which is sixteen times sixteen times twenty thousand
        // samples at the widest — so the *ranking* is done on a stride. A
        // correlation estimated on every eighth sample of a passage is the
        // same correlation; the weights that go on the wire are fitted on all
        // of it, further down.
        let stride = (interval[dest].len() / RANK_SAMPLES).max(1);
        let target: Vec<i32> = interval[dest].iter().step_by(stride).copied().collect();
        let frames = target.len();
        if frames == 0 {
            return Vec::new();
        }
        let mut left = vec![0.0f64; frames];
        for (slot, sample) in left[..frames].iter_mut().zip(&target) {
            *slot = f64::from(*sample);
        }
        let strided: Vec<Vec<i32>> = (0..=available)
            .map(|source| interval[source].iter().step_by(stride).copied().collect())
            .collect();

        let mut chosen: Vec<usize> = Vec::new();
        while chosen.len() < matrix::MAX_FIT_SOURCES {
            let own: f64 = left[..frames].iter().map(|v| v * v).sum();
            if own <= 0.0 {
                break;
            }
            let mut best = None;
            let mut best_score = 0.0f64;
            for (source, other) in strided.iter().enumerate() {
                if source == dest || chosen.contains(&source) {
                    continue;
                }
                if other.len() != frames {
                    continue;
                }
                let theirs = other
                    .iter()
                    .map(|v| f64::from(*v) * f64::from(*v))
                    .sum::<f64>();
                if theirs <= 0.0 {
                    continue;
                }
                let together: f64 = left[..frames]
                    .iter()
                    .zip(other.iter())
                    .map(|(a, b)| a * f64::from(*b))
                    .sum();
                // The share of what is left that this source explains, which
                // is the correlation squared and needs no square root to
                // compare.
                let score = together * together / (own * theirs);
                if score > best_score {
                    best_score = score;
                    best = Some(source);
                }
            }
            let Some(source) = best else { break };
            chosen.push(source);

            // What the fit so far leaves, which is what the next source is
            // judged on. In the decoder's arithmetic — one accumulator for
            // every source and one shift at the end — because that is the
            // signal the coding would actually see.
            let windows: Vec<&[i32]> = chosen
                .iter()
                .map(|source| strided[*source].as_slice())
                .collect();
            let Some(weights) = matrix::best_weights(&target, &windows, matrix::FRACTION) else {
                chosen.pop();
                break;
            };
            for (frame, slot) in left[..frames].iter_mut().enumerate() {
                let mut sum = 0i64;
                for (window, weight) in windows.iter().zip(&weights) {
                    sum += i64::from(*weight) * i64::from(window[frame]);
                }
                *slot = f64::from(target[frame] - (sum >> matrix::FRACTION) as i32);
            }
        }
        chosen
    }

    /// Try coding `dest` against one, two and three of its partners, and keep
    /// the cheapest that beats coding it alone.
    ///
    /// One direction only, and the caller picks it. Trying both — which this
    /// did while the pairs were always inside one substream, where either way
    /// round is legal — writes a matrix whose source the substream declaring
    /// it has not decoded. That parses, every checksum verifies, and a decoder
    /// stopping at the six-channel presentation hands back a channel still
    /// matrixed.
    fn judge_matrix(&self, dest: usize, scratch: &mut [i32]) -> Judged {
        let frames = self.coded.frame_size;
        // A matrix is described once in every substream that can reach its
        // destination — the one that codes the channel, and every presentation
        // behind it, each of which has to undo the matrix for itself. All of
        // them are paid for, so all of them are costed. Which is also why a
        // matrix on the stereo pair has to earn four descriptions and one on
        // the last substream only one.
        let copies = self.coded.ranges[..self.coded.substreams]
            .iter()
            .filter(|range| range.last >= dest)
            .count();
        let last = self.coded.ranges[self.coded.substreams - 1];

        // 🔴 Judged on units spread over the interval the matrix will stand
        // for, not on its first unit alone. A description costs the better
        // part of two hundred bits with its copies, and forty samples of a
        // channel a matrix takes three bits off gain a hundred and twenty —
        // so against one unit no matrix on a quiet channel ever paid, though
        // it stood for a hundred and twenty-eight of them. Measured: four
        // matrices chosen in sixteen intervals of a scene whose stored
        // channels the covariance said held ten bits of prediction. The
        // interval is held; it can be looked at. Each unit is priced as a
        // restarting one, which overstates the first eight samples the same
        // way for every candidate and for coding the channel alone.
        let units = (self.prepared[dest].len() / frames).max(1);
        let trials: Vec<usize> = (0..TRIAL_UNITS.min(units))
            .map(|which| which * units / TRIAL_UNITS.min(units))
            .collect();

        // What coding it alone costs, which is the yardstick.
        let mut plain = 0usize;
        for unit in &trials {
            let at = unit * frames;
            let (context, samples) = if *unit == 0 {
                (self.past[dest].window(), &self.prepared[dest][..frames])
            } else {
                (
                    &self.prepared[dest][..at],
                    &self.prepared[dest][at..at + frames],
                )
            };
            plain = plain.saturating_add(restart_cost(context, samples, scratch));
        }

        let mut best: Option<(Primitive, usize)> = None;
        let mut refused = 0u64;
        let domain = self.domain_of(dest);
        let mut matrixed: Vec<i32> = Vec::with_capacity(self.prepared[dest].len());
        for index in 0..self.candidates[dest].len() {
            let Some(candidate) = Primitive::fit(
                dest,
                &self.candidates[dest][index].sources,
                &self.candidates[dest][index].weights,
                matrix::FRACTION,
            ) else {
                continue;
            };

            // The decoder accumulates every source at the matrix
            // accumulator's own scale and shifts once at the end, so the
            // inverse has to as well: shifting each term on its own rounds
            // differently and the stream decodes to something else.
            matrixed.clear();
            for frame in 0..self.prepared[dest].len() {
                let mut sum = 0i64;
                for (source, weight) in self.candidates[dest][index]
                    .sources
                    .iter()
                    .zip(&self.candidates[dest][index].weights)
                {
                    sum += i64::from(*weight) * i64::from(self.prepared[*source][frame]);
                }
                matrixed.push(self.prepared[dest][frame] - (sum >> matrix::FRACTION) as i32);
            }

            // 🔴 And what it leaves has to fit the domain the decoder
            // reconstructs it in — see `domain_of`. A decoder rebuilds the
            // matrixed channel, residual plus prediction, in that many bits
            // and refuses the stream past them, which the reference reports
            // as a saturated recorrelator. The weight is a least-squares fit
            // over the interval, so it takes energy off on average and not on
            // every sample: on a unit where the sources swing the other way
            // it adds, and a leading channel holding a fold amplified by its
            // host's share has no room for that. The cascade is checked
            // against the domain before it is kept; this is the same check on
            // the step that runs after it. Measured on a re-voiced programme:
            // fifty-four saturations in the first two substreams, none once
            // these were refused — three candidates turned away and seven
            // decided matrices dropped over its first thirty-one minutes, for
            // a stream two kilobytes smaller.
            if matrixed
                .iter()
                .any(|sample| !(-domain..domain).contains(sample))
            {
                if std::env::var_os("HZ_DOMAIN").is_some() {
                    let worst = matrixed.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
                    let count = matrixed
                        .iter()
                        .filter(|s| !(-domain..domain).contains(s))
                        .count();
                    let peak = self.prepared[dest]
                        .iter()
                        .map(|s| s.unsigned_abs())
                        .max()
                        .unwrap_or(0);
                    eprintln!(
                        "domain: dest {dest} sources {:?} weights {:?} worst {worst} over {count} of {} (dest peak {peak})",
                        self.candidates[dest][index].sources,
                        self.candidates[dest][index].weights,
                        matrixed.len()
                    );
                }
                refused += 1;
                continue;
            }

            // The description stands for the whole interval and the trial
            // reads part of it, so it is charged in the same proportion — in
            // full against four units it was a bit a sample of gain that
            // never paid, and a stereo pair's mid/side matrix is worth a
            // fifth of that.
            let description = crate::rate::matrix(candidate.cost_bits(&last), copies);
            // 🔴 And a margin. The winner among a destination's candidates
            // is picked on four units of a hundred and twenty-eight, so it is
            // the one the sample favoured as much as the one that is best,
            // and a matrix that wins its trials narrowly loses the interval.
            // Half a bit a sample on the trials is what that luck is worth,
            // measured: without it the stream is 0.3 to 0.8 % larger on
            // every programme fixture, and on none of them smaller.
            let mut cost =
                (description * trials.len()).div_ceil(units) + MATRIX_MARGIN * trials.len();
            for unit in &trials {
                let at = unit * frames;
                let context = if *unit == 0 {
                    self.past[dest].window()
                } else {
                    &matrixed[..at]
                };
                cost =
                    cost.saturating_add(restart_cost(context, &matrixed[at..at + frames], scratch));
                if cost >= plain {
                    break;
                }
            }
            if cost < plain && best.as_ref().is_none_or(|(_, before)| cost < *before) {
                best = Some((candidate, cost));
            }
        }

        Judged {
            best: best.map(|(candidate, _)| candidate),
            refused,
        }
    }
}

impl Encoder {
    /// Give each matrix the step its own interval measured.
    ///
    /// The steps were found by [`Encoder::line_through`], which fits the
    /// weight on four slices of the interval and puts a line through them —
    /// so a matrix that ramps states where it starts and how fast it moves,
    /// both measured on the passage it will stand for. This used to
    /// extrapolate from the interval *before*, because the encoder wrote each
    /// unit as it arrived and had nothing else to go on.
    ///
    /// Only the last substream's syntax can say a step, and only for a
    /// destination no earlier substream describes: a matrix stated twice has
    /// to be stated the same way both times, or a decoder stopping at the
    /// earlier presentation reconstructs different audio. That rule is
    /// applied where the steps are measured, so a matrix that may not move has
    /// none to copy here.
    fn aim_the_matrices(&mut self) {
        let mut aimed = 0;
        for matrix in &mut self.matrices {
            let Some(candidate) = self.candidates[matrix.dest]
                .iter()
                .find(|candidate| candidate.matches(matrix))
            else {
                continue;
            };
            let mut moved = false;
            for (source, step) in candidate.sources.iter().zip(&candidate.steps) {
                // Stated whenever the weight moved at all. A threshold was
                // tried — a sixty-fourth of unity over the interval, to skip
                // steps too small to pay for themselves — and it cost 184
                // bytes on the sweeping fixture to save six on the still one.
                //
                // 🔴 But never on a source whose coefficient is zero. The
                // stream names its sources by a mask of the non-zero
                // coefficients and carries a step for those alone, so a step
                // on a weight that starts at nothing is one the decoder never
                // sees — while this encoder's own arithmetic applied it. A
                // weight that crosses zero inside the interval fits a line
                // through zero, and on the dense channels a fold leaves it
                // does so often enough that two intervals of a programme's
                // first four minutes decoded to the wrong samples with every
                // checksum right and the lossless check saying so.
                if *step != 0 && matrix.coefficients[*source] != 0 {
                    matrix.delta[*source] = *step;
                    moved = true;
                }
            }
            if moved {
                aimed += 1;
            }
        }
        self.stats.aimed_matrices += aimed;
    }
}

/// How many units of an interval a matrix candidate is judged on, spread
/// evenly over it. Four: the search is the whole filter search over each
/// unit, and four of them took the encode of a five-minute programme from 17 s
/// to 19 s while finding what one unit could not.
const TRIAL_UNITS: usize = 4;

/// Bits a matrix has to win each trial unit by, over coding the channel
/// alone: half a bit a sample of a unit at the base rate. Measured over the
/// fixture set at 0, 40, 80 and 160 over the four trials: 80 is the best on
/// every programme fixture, 40 and 160 both a little behind it, and the
/// synthetic fixtures do not move.
const MATRIX_MARGIN: usize = 20;

/// What a candidate costs, priced the way a restarting access unit is written.
///
/// A matrix is only ever chosen in a unit that restarts, and such a unit is
/// two blocks: the first unpredicted, the second predicted from the state the
/// first leaves. Pricing it as one predicted block instead — which is what
/// this did before the restart block existed — overstates what prediction can
/// do for the first eight samples and so understates every candidate equally.
/// Equally is not harmless: the matrix and the plain coding respond to it
/// differently, and the decision stands for a whole restart interval.
///
/// Three estimators were measured over six fixtures — this one, pricing the
/// unit as a single block from a zero state, and pricing it as a single block
/// from the state a *typical* unit of the interval carries. This one wins or
/// ties on five of the six, by up to 3.6 kB; the zero-state one wins on one,
/// a programme in stereo, by 680 bytes. Pricing what is actually written is the
/// tie-breaker.
fn restart_cost(context: &[i32], samples: &[i32], scratch: &mut [i32]) -> usize {
    let (_, head) = filter::plain(&samples[..RESTART_BLOCK], &mut scratch[..RESTART_BLOCK]);
    let state = filter::state_from(&samples[..RESTART_BLOCK], &scratch[..RESTART_BLOCK]);
    let (_, tail) = filter::choose(
        context,
        state,
        &samples[RESTART_BLOCK..],
        &mut scratch[RESTART_BLOCK..],
        &filter::Filter::default(),
    );
    head.saturating_add(tail)
}

/// The format's way of folding a 32-bit accumulator into the 8 bits it stores.
fn xor_to_byte(mut value: u32) -> u8 {
    value ^= value >> 16;
    value ^= value >> 8;
    value as u8
}

/// The end-of-stream marker, as the two 16-bit fields it is written as.
pub(crate) fn end_of_stream(shorten_by: u16) -> [u32; 2] {
    // The high bits mark it as a TrueHD shortening rather than a plain end of
    // stream; the low thirteen are the count.
    [END_OF_STREAM, u32::from(shorten_by & 0x1fff) | 0xe000]
}

/// What an interval's restart header will describe, decided from the interval
/// alone: its shifts, the cascade that puts its folds in the channels, and the
/// rows each early substream declares.
///
/// A thing of its own so that the next interval's can be worked out while this
/// one is being written. It reads nothing the writing changes — what was asked
/// for, the shape of the stream, and the interval's own samples — and the
/// encoder takes what it decided just before writing the interval it is for.
#[derive(Debug)]
struct Planning {
    wants: Vec<crate::hierarchy::Presentation>,
    channels: usize,
    coded: FrameSize,
    decorrelation: bool,
    threads: std::sync::Arc<Threads>,
    interval_shift: Option<Vec<u8>>,
    arrangement: Vec<Primitive>,
    presentations: Vec<Vec<Primitive>>,
    assignment: Vec<Vec<u8>>,
    presentation_shift: Vec<Vec<u8>>,
    permutation: Vec<usize>,
    stats: Stats,
    /// How long deciding took, for `HZ_TIME`.
    took: std::time::Duration,
}

/// The one thing about the stream's shape [`Planning`] reads.
#[derive(Debug, Clone, Copy)]
struct FrameSize {
    frame_size: usize,
}

/// An interval that is complete and not yet written: its units, what was
/// decided for it, and the dynamic range the caller had asked for when it was
/// complete — which is what writing it at once would have stated.
#[derive(Debug)]
struct Deferred {
    held: Vec<Held>,
    planning: Planning,
    dynamic_range: [Option<DynamicRange>; MAX_SUBSTREAMS],
}

impl Planning {
    /// Decide an interval.
    fn plan(mut self, held: &[Held]) -> Self {
        let start = std::time::Instant::now();
        self.settle_the_shifts(held);
        self.build_the_presentations(held);
        self.took = start.elapsed();
        self
    }

    /// The shifts this interval will hold, or none where it holds none.
    ///
    /// The smallest any of its units offers, per channel, because a shift is
    /// only free if every unit really has those bits — take one unit's four
    /// and a later unit with two loses two of the programme.
    fn settle_the_shifts(&mut self, held: &[Held]) {
        self.interval_shift = None;
        if self.wants.is_empty() {
            return;
        }
        let mut shifts = vec![crate::frame::MAX_OUTPUT_SHIFT; self.channels];
        let mut said = vec![false; self.channels];
        for unit in held {
            for (channel, shift) in shifts.iter_mut().enumerate() {
                let mut bits = 0u32;
                for frame in 0..unit.frames {
                    let sample = unit.samples[frame * self.channels + channel];
                    if sample != 0 {
                        bits |= sample.unsigned_abs();
                    }
                }
                if bits != 0 {
                    said[channel] = true;
                    *shift = (*shift)
                        .min((bits.trailing_zeros() as u8).min(crate::frame::MAX_OUTPUT_SHIFT));
                }
            }
        }
        // A channel that is silent all interval has nothing to say about it.
        for (channel, shift) in shifts.iter_mut().enumerate() {
            if !said[channel] {
                *shift = 0;
            }
        }
        self.interval_shift = Some(shifts);
    }

    /// Build the arrangement and the rows each substream declares.
    ///
    /// 🔴 In the units the shifts make. A channel shifted down by three and one
    /// shifted down by four are not the same units, so a row mixing them has to
    /// carry the ratio — which is what the reference's own rows do, and what
    /// gives its amplifying matrices the headroom they need. Written over the
    /// elements, presentation `j`'s row wants `2^(s_e - s_j)` of element `e`.
    ///
    /// Everything is cleared where it cannot be done, so the stream is written
    /// without folds rather than written wrong.
    fn build_the_presentations(&mut self, units: &[Held]) {
        self.arrangement.clear();
        self.presentations.clear();
        self.assignment.clear();
        self.presentation_shift.clear();
        self.permutation = (0..self.channels).collect();
        if self.wants.is_empty() {
            return;
        }
        self.stats.folds_asked += 1;
        let Some(shifts) = self.interval_shift.clone() else {
            return;
        };

        // 🔴 Per channel first, and a common shift if that cannot be written.
        //
        // A row whose destination is shifted by nothing and whose sources are
        // shifted by four has to ask for sixteen times what it wants, and the
        // field holds under two — measured at 136 on a real programme. The
        // reference does not have that problem because it **restates the
        // shifts in every substream**, so each presentation is written in units
        // that suit it: its stereo substream says `[3, 4]` for the same two
        // channels its 5.1 substream says `[3, 2]` for.
        //
        // This encoder states one set for the whole unit, so where the shifts
        // spread too far to write it takes the smallest of them for every
        // channel — which costs the dead bits the others had, and is what a
        // programme with one unrounded channel among rounded ones ends up
        // paying until the shifts are stated per substream.
        if let Ok(built) = self.try_to_build(units, &shifts) {
            self.take_what_was_built(built);
            return;
        }
        let least = shifts
            .iter()
            .take(self.channels)
            .copied()
            .min()
            .unwrap_or(0);
        let common = vec![least; shifts.len()];
        match self.try_to_build(units, &common) {
            Ok(built) => {
                self.interval_shift = Some(common);
                self.take_what_was_built(built);
            }
            Err(Refusal::OverTheDomain) => self.stats.folds_over_the_domain += 1,
            Err(Refusal::Unwritable) => {}
        }
    }

    /// Keep what a build produced.
    fn take_what_was_built(&mut self, built: Built) {
        self.stats.folds_carried += 1;
        if std::env::var_os("HZ_FOLD").is_some() {
            eprintln!(
                "fold: built, {} steps, presentations {:?} rows, stated shifts {:?}",
                built.steps.len(),
                built.rows.iter().map(Vec::len).collect::<Vec<_>>(),
                built
                    .stated
                    .iter()
                    .map(|s| s.first().copied())
                    .collect::<Vec<_>>()
            );
        }
        self.arrangement = built.steps;
        self.presentations = built.rows;
        self.assignment = built.assignment;
        self.presentation_shift = built.stated;
        // The elements go into the internal channels the cascade was built
        // for, the shifts settled per element follow them there, and the last
        // substream says which output each internal channel is so a full
        // decode hands the elements back in order.
        if let Some(shifts) = &self.interval_shift {
            let permuted: Vec<u8> = built
                .element_of
                .iter()
                .map(|element| shifts.get(*element).copied().unwrap_or(0))
                .collect();
            self.interval_shift = Some(permuted);
        }
        self.assignment.push(
            built
                .element_of
                .iter()
                .map(|element| *element as u8)
                .collect(),
        );
        self.permutation = built.element_of;
    }

    /// The arrangement and the declared rows, in the units these shifts make.
    ///
    /// Written over the elements, a presentation's row wants `2^s[e]` of
    /// element `e` — a channel shifted down by four is in units sixteen times
    /// smaller, so a row that wants a quarter of it has to ask for four. The
    /// destination's own shift is the other half of the ratio and belongs to
    /// whichever internal channel ends up computing the row, which the
    /// factorisation decides, so it goes in there.
    fn try_to_build(&self, units: &[Held], shifts: &[u8]) -> Result<Built, Refusal> {
        let elements = self.channels;
        let scaled: Vec<crate::hierarchy::Presentation> = self
            .wants
            .iter()
            .map(|presentation| crate::hierarchy::Presentation {
                channels: presentation.channels,
                rows: presentation
                    .rows
                    .iter()
                    .map(|row| {
                        row.iter()
                            .enumerate()
                            .map(|(element, gain)| {
                                gain * f64::from(shifts.get(element).copied().unwrap_or(0)).exp2()
                            })
                            .collect()
                    })
                    .collect(),
            })
            .collect();

        // How loud each element is over the interval, as bits: what a channel
        // holding it raw costs, and half of what decides where it goes.
        let loudness: Vec<f64> = (0..elements)
            .map(|element| {
                let mut sum = 0.0f64;
                let mut count = 0usize;
                for unit in units {
                    for frame in 0..unit.frames {
                        let sample = f64::from(
                            unit.samples[frame * self.channels + element]
                                >> shifts.get(element).copied().unwrap_or(0),
                        );
                        sum += sample * sample;
                        count += 1;
                    }
                }
                (sum / count.max(1) as f64).sqrt().max(1.0).log2()
            })
            .collect();

        // Which element goes in which internal channel — see
        // [`crate::arrange::place_elements`] for what it optimises. The
        // hierarchy is built once to learn its shape, which channels are
        // slots no fold row takes among it; the matching then decides every
        // channel's element at once, hosts and slots together; and the
        // hierarchy is built again with the slots' elements first, so that
        // is what fills them. Everything below is over internal channels.
        let sketch = crate::hierarchy::build(&scaled, elements).ok_or(Refusal::Unwritable)?;
        let channel_of = crate::arrange::place_elements(
            &sketch.rows,
            &sketch.within,
            elements,
            &loudness,
            &sketch.filled,
        );
        let mut element_of = vec![usize::MAX; elements];
        for (element, channel) in channel_of.iter().enumerate() {
            if *channel >= elements || element_of[*channel] != usize::MAX {
                return Err(Refusal::Unwritable);
            }
            element_of[*channel] = element;
        }
        let mut prefer: Vec<usize> = (0..elements)
            .filter(|channel| sketch.filled.get(*channel).copied().unwrap_or(false))
            .map(|channel| element_of[channel])
            .collect();
        let rest: Vec<usize> = (0..elements)
            .filter(|element| !prefer.contains(element))
            .collect();
        prefer.extend(rest);
        let built = crate::hierarchy::build_preferring(&scaled, elements, &prefer)
            .ok_or(Refusal::Unwritable)?;
        if built.within != sketch.within || built.filled != sketch.filled {
            return Err(Refusal::Unwritable);
        }
        let rows: Vec<Vec<f64>> = built
            .rows
            .iter()
            .map(|row| crate::arrange::through(row, &channel_of))
            .collect();
        let internal: Vec<crate::hierarchy::Presentation> = scaled
            .iter()
            .map(|presentation| crate::hierarchy::Presentation {
                channels: presentation.channels,
                rows: presentation
                    .rows
                    .iter()
                    .map(|row| crate::arrange::through(row, &channel_of))
                    .collect(),
            })
            .collect();

        let fold_log = std::env::var_os("HZ_FOLD").is_some();
        let Some(cascade) = crate::arrange::arrange(&rows, &built.within, elements) else {
            if fold_log {
                eprintln!("fold: no cascade hosts the rows, shifts {shifts:?}");
            }
            return Err(Refusal::Unwritable);
        };
        if fold_log {
            eprintln!(
                "fold: loudness {:?}\n      element_of {:?}\n      hosts {:?} scales {:?}\n      {} steps, amplification {:.2}x",
                loudness
                    .iter()
                    .map(|l| (l * 10.0).round() / 10.0)
                    .collect::<Vec<_>>(),
                element_of,
                cascade.hosts,
                cascade
                    .scales
                    .iter()
                    .map(|s| (s * 1000.0).round() / 1000.0)
                    .collect::<Vec<_>>(),
                cascade.steps.len(),
                cascade.amplification()
            );
        }
        // 🔴 And whether the coded domain can hold it, which is **run** rather
        // than reasoned about. A cascade stores more than it was given — see
        // [`crate::arrange::Arrangement::amplification`] — and a decoder
        // enforces twenty-four bits by refusing the stream outright.
        //
        // The dead bits are the headroom and each channel has its own, so what
        // a channel really stores is what the arithmetic makes of this
        // interval's own samples. Bounding it instead means bounding a sum of
        // magnitudes, which assumes every element peaks at once and in step:
        // measured at 3.16x full scale on a programme whose cascade fits with
        // room to spare. The encoder is holding the samples; it can just look.
        if !self.the_cascade_fits(units, shifts, &element_of, &cascade.steps) {
            if fold_log {
                eprintln!("fold: the cascade does not fit the codec's domain");
            }
            return Err(Refusal::OverTheDomain);
        }

        // How loud each early presentation's fold gets over the interval, in
        // the samples a decoder hands back — what a presentation's shifts
        // have to leave room for.
        let early = &self.wants[..self.wants.len().saturating_sub(1)];
        let rows: Vec<&Vec<f64>> = early.iter().flat_map(|p| p.rows.iter()).collect();
        let mut loudest = self
            .threads
            .map(&rows, |row| {
                let mut peak = 0.0f64;
                for unit in units {
                    for frame in unit.samples.chunks_exact(elements).take(unit.frames) {
                        let value: f64 = row
                            .iter()
                            .zip(frame)
                            .map(|(gain, sample)| gain * f64::from(*sample))
                            .sum();
                        peak = peak.max(value.abs());
                    }
                }
                peak
            })
            .into_iter();
        let peaks: Vec<Vec<f64>> = early
            .iter()
            .map(|presentation| loudest.by_ref().take(presentation.rows.len()).collect())
            .collect();
        let Some(written) = Encoder::presentation_rows(
            &self.threads,
            &cascade.held,
            &internal,
            elements,
            fold_log,
            false,
            &peaks,
        ) else {
            return Err(Refusal::Unwritable);
        };
        let (steps, (rows, assignment, stated)) = self.decorrelate(
            units,
            shifts,
            &element_of,
            cascade.steps,
            cascade.held,
            &internal,
            written,
            &peaks,
            fold_log,
        );
        Ok(Built {
            steps,
            rows,
            assignment,
            stated,
            element_of,
        })
    }

    /// Does everything this interval stores stay inside the codec's domain?
    ///
    /// The samples as they will really be written: shifted down by what the
    /// interval settled on, then with the cascade taken off. Anything outside
    /// twenty-four bits is a stream a decoder refuses, so the arrangement is
    /// dropped and the interval written without folds.
    fn the_cascade_fits(
        &self,
        units: &[Held],
        shifts: &[u8],
        element_of: &[usize],
        steps: &[Primitive],
    ) -> bool {
        let width = self.coded.frame_size;
        // Each unit on its own: nothing carries from one to the next, and the
        // answer is whether every one of them fits, which no order changes.
        let fits = |unit: &Held| {
            let mut scratch: Vec<Vec<i32>> = vec![vec![0; width]; self.channels];
            for (channel, plane) in scratch.iter_mut().enumerate() {
                let element = element_of.get(channel).copied().unwrap_or(channel);
                let shift = shifts.get(element).copied().unwrap_or(0);
                for (frame, slot) in plane.iter_mut().enumerate().take(width) {
                    let sample = if frame < unit.frames {
                        unit.samples[frame * self.channels + element]
                    } else {
                        0
                    };
                    *slot = sample >> shift;
                }
            }
            for step in steps.iter().rev() {
                crate::arrange::unapply_within(step, &mut scratch, width);
            }
            !scratch.iter().any(|plane| {
                plane
                    .iter()
                    .any(|sample| !(-DOMAIN..DOMAIN).contains(sample))
            })
        };
        self.threads.map(units, fits).into_iter().all(|fit| fit)
    }

    /// Take out of each early channel what the channels below it predict of
    /// it, where the presentations can still be written over what is left.
    ///
    /// # Why
    ///
    /// The hierarchy's channels overlap: the 5.1's centre is also in the
    /// stereo pair, the 7.1's backs are in the 5.1's surrounds, and every one
    /// of them is a mix of the same elements. Stored as they are, the codec
    /// pays for the shared part twice. Coding matrices used to take it out —
    /// until FFmpeg's limit of eight matrices a substream under restart sync
    /// words A and B, which the 7.1's eight rows already fill, left them no
    /// room in channels 0 to 7. That cost folded streams five per cent.
    ///
    /// The reference pays nothing for it, because its channels are stored
    /// already decorrelated and the correction lives in rows it declares
    /// anyway. This does the same: a lifting step per channel, in the last
    /// substream's cascade, which has room, and the early substreams' rows
    /// rewritten over the decorrelated channels, at no extra rows.
    ///
    /// # How
    ///
    /// Channel `k`, from one up to the widest early presentation, is fitted by
    /// least squares over the interval to channels `0..k` as the cascade
    /// stores them, and a step `x[k] += Σ w·x[j]` goes in front of the cascade
    /// where the fit takes out more than [`DECORRELATION_BITS`]. A decoder
    /// runs these steps first and in rising `k`, each reading channels already
    /// restored; the encoder undoes them last and in falling `k`, each reading
    /// channels it has not touched yet — the same sums either way, so the
    /// round trip is exact.
    ///
    /// # Why one at a time
    ///
    /// A channel with its prediction taken out holds a small remainder, and
    /// the rows that rebuild a presentation from it then ask for coefficients
    /// far past the field's two: measured at 30 to 130, mostly in the 5.1, and
    /// the 7.1 has no spare row to scale one up with. Taken all at once, the
    /// steps left the rows writable in one interval in seven, for 0.5 % of the
    /// stream. So each step is kept only if every presentation can still be
    /// written after it, and the rest are skipped.
    ///
    /// What made most of them writable is a shift per presentation channel —
    /// see [`crate::hierarchy::rows_over_choosing_shifts`] — which says a row
    /// asking for a hundred at a hundred-and-twenty-eighth of its size. With a
    /// shift for every channel alike this came to 1.56 % of the programme
    /// slice; with one per channel, and held to the precision and the room
    /// at full scale a fold needs, 2.7 %, against 5.5 % had every step been
    /// writable.
    ///
    /// # Measured and put down
    ///
    /// - **Only from the channel's own substream**, which keeps the rows
    ///   simpler: +0.88 %, worse than nothing.
    /// - **A prediction kept orthogonal, over the elements, to what the
    ///   channel holds**, so that it keeps its size and the rows stay small:
    ///   no gain at all. The saving *is* the shared elements, which is what
    ///   makes the remainder small; the two cannot be had apart.
    /// - **The threshold** is not the modelled gain it looks like: the raw
    ///   energy predicts the coded cost badly, and a step it rates at a few
    ///   hundred bits can cost more than it saves. On the slice, with a shift
    ///   per channel and before the precision was held: 500 bits 1.6 %, 2 000
    ///   3.0 %, **5 000 4.0 %**, 10 000 3.7 %, 20 000 2.8 %.
    #[allow(clippy::too_many_arguments)]
    fn decorrelate(
        &self,
        units: &[Held],
        shifts: &[u8],
        element_of: &[usize],
        steps: Vec<Primitive>,
        held: Vec<Vec<f64>>,
        internal: &[crate::hierarchy::Presentation],
        written: PresentationRows,
        peaks: &[Vec<f64>],
        fold_log: bool,
    ) -> (Vec<Primitive>, PresentationRows) {
        let upto = internal
            .len()
            .checked_sub(2)
            .map_or(0, |widest| internal[widest].channels.min(self.channels));
        let room = DECORRELATING_STEPS_AT_MOST.saturating_sub(steps.len());
        if !self.decorrelation || upto < 2 || room == 0 {
            return (steps, written);
        }

        // The Gram matrix of the stored channels over the interval: what each
        // unit stores worked out on its own, then each entry summed on its
        // own, over the units and frames in order. Each entry is the same
        // additions in the same order as one loop over all of them made, so
        // the same to the bit — and there are thirty-six of them, where the
        // loop was one thread and the longest thing a fold's building did.
        let width = self.coded.frame_size;
        let stored: Vec<Vec<Vec<i32>>> = self.threads.map(units, |unit| {
            let mut scratch: Vec<Vec<i32>> = vec![vec![0; width]; self.channels];
            for (channel, plane) in scratch.iter_mut().enumerate() {
                let element = element_of.get(channel).copied().unwrap_or(channel);
                let shift = shifts.get(element).copied().unwrap_or(0);
                for (frame, slot) in plane.iter_mut().enumerate() {
                    *slot = if frame < unit.frames {
                        unit.samples[frame * self.channels + element] >> shift
                    } else {
                        0
                    };
                }
            }
            for step in steps.iter().rev() {
                crate::arrange::unapply_within(step, &mut scratch, width);
            }
            scratch.truncate(upto);
            for plane in &mut scratch {
                plane.truncate(unit.frames);
            }
            scratch
        });
        let samples: usize = units.iter().map(|unit| unit.frames).sum();
        let pairs: Vec<(usize, usize)> = (0..upto)
            .flat_map(|i| (0..=i).map(move |j| (i, j)))
            .collect();
        let sums = self.threads.map(&pairs, |&(i, j)| {
            let mut sum = 0.0f64;
            for unit in &stored {
                for (a, b) in unit[i].iter().zip(&unit[j]) {
                    sum += f64::from(*a) * f64::from(*b);
                }
            }
            sum
        });
        let mut gram = vec![vec![0.0f64; upto]; upto];
        for (&(i, j), sum) in pairs.iter().zip(sums) {
            gram[i][j] = sum;
        }
        for i in 0..upto {
            for j in 0..i {
                gram[j][i] = gram[i][j];
            }
        }

        let mut decorrelating = Vec::new();
        let mut holds = held.clone();
        let mut best = None;
        for k in 1..upto {
            if decorrelating.len() >= room {
                break;
            }
            let before = gram[k][k];
            if before <= 0.0 {
                continue;
            }
            let Some(weights) = least_squares(&gram, k) else {
                continue;
            };
            let explained: f64 = weights.iter().zip(&gram[k][..k]).map(|(w, g)| w * g).sum();
            let after = (before - explained).max(before * 1e-9);
            if 0.5 * (before / after).log2() * (samples as f64) < DECORRELATION_BITS {
                continue;
            }
            let mut gains = vec![0.0f64; self.channels];
            gains[k] = 1.0;
            gains[..k].copy_from_slice(&weights);
            let Some(step) = Primitive::rounded(k, &gains, crate::matrix::FRACTION) else {
                continue;
            };
            // What the channel holds once the step's prediction is taken out,
            // at the coefficients the stream will carry — against what each
            // source holds *before* any step, since the step reads it
            // restored.
            let mut remainder = holds[k].clone();
            for (j, source) in held.iter().enumerate().take(k) {
                let w = f64::from(step.coefficients[j]) / f64::from(crate::matrix::UNITY);
                if w != 0.0 {
                    for (into, from) in remainder.iter_mut().zip(source) {
                        *into -= w * from;
                    }
                }
            }
            let kept = std::mem::replace(&mut holds[k], remainder);
            match Encoder::presentation_rows(
                &self.threads,
                &holds,
                internal,
                self.channels,
                false,
                true,
                peaks,
            ) {
                Some(rows) => {
                    decorrelating.push(step);
                    best = Some(rows);
                }
                None => holds[k] = kept,
            }
        }
        let Some(rows) = best else {
            return (steps, written);
        };

        // And whether the codec's domain holds what they leave, which is run
        // rather than reasoned about for the same reason the cascade's is.
        let count = decorrelating.len();
        let mut candidate = decorrelating;
        candidate.extend_from_slice(&steps);
        if !self.the_cascade_fits(units, shifts, element_of, &candidate) {
            if fold_log {
                eprintln!("fold: decorrelating {count} channels leaves them over the domain");
            }
            return (steps, written);
        }
        if fold_log {
            eprintln!("fold: {count} channels decorrelated");
        }
        (candidate, rows)
    }
}

/// An interval prepared and not yet decided block by block: what the writing
/// will need, and what the block decisions read that the preparing of the
/// next interval moves on — the past before this interval, and each
/// channel's second filter.
#[derive(Debug)]
struct Front {
    record: Record,
    past: Vec<History>,
    second_mode: Vec<Option<usize>>,
    /// The one first-filter order to fit, where the effort fixes it.
    order: Option<usize>,
}

/// What decides each block of an interval, channel by channel.
///
/// A channel's blocks are decided one after another, each from the filter the
/// one before it left, so an interval keeps as many threads busy as it has
/// channels and no more. Taken out of the encoder so that it runs beside the
/// preparing of the next interval, which fills the rest: it reads the
/// interval's prepared samples, the past before it and its second filters,
/// all handed over in a [`Front`], and nothing the preparing changes.
#[derive(Debug)]
struct IntervalDecider {
    channels: usize,
    coded: Coded,
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    threads: std::sync::Arc<Threads>,
    /// The whole interval's per-channel decisions, made in one pass before any
    /// of it is written — see [`Encoder::decide_interval`]. Indexed
    /// `[channel][unit]`, with the restarting unit's first block apart.
    interval_decided: Vec<Vec<Decided>>,
    /// And its residuals, `[channel]` by unit, laid out like `prepared`.
    interval_planes: Vec<Vec<i32>>,
    interval_restart: Vec<Decided>,
    /// The last residuals each channel was coded with, most recent last. A
    /// decoder carries them as the second filter's state, so this encoder has
    /// to as well — and they are the residuals of whatever filter was actually
    /// chosen, not of any candidate.
    past_residuals: Vec<Vec<i32>>,
    prepared: Vec<Vec<i32>>,
    past: Vec<History>,
    second_mode: Vec<Option<usize>>,
    /// What deciding the blocks counts; see [`Encoder::stats`].
    stats: Stats,
}

impl IntervalDecider {
    /// Decide an interval's blocks, and hand over the record for writing.
    fn decide(&mut self, front: Front) -> (Record, std::time::Duration) {
        let start = std::time::Instant::now();
        let Front {
            mut record,
            past,
            second_mode,
            order,
        } = front;
        let units = record.held.len();
        self.prepared = std::mem::take(&mut record.prepared);
        self.past = past;
        self.second_mode = second_mode;
        self.decide_interval(units, order);
        let channels = self.channels;
        record.prepared = std::mem::take(&mut self.prepared);
        record.interval_decided =
            std::mem::replace(&mut self.interval_decided, vec![Vec::new(); channels]);
        record.interval_planes =
            std::mem::replace(&mut self.interval_planes, vec![Vec::new(); channels]);
        record.interval_restart = self.interval_restart.clone();
        (record, start.elapsed())
    }

    /// Decide every channel of the whole interval, in one pass.
    ///
    /// # Why this is not done unit by unit
    ///
    /// It used to be: each access unit forked its channels across the pool and
    /// joined them again. Forty samples of work a channel is less than the
    /// fork costs, and over a five-minute programme that is 360 000 forks —
    /// the phase scaled 3.2× on twelve threads where the second-filter search,
    /// which forks once an interval, scaled 7.7×.
    ///
    /// So the fork moves out here: one per interval, and each thread walks its
    /// own channel through all 128 units. That is allowed because a channel's
    /// search reads only that channel — its samples, its history, its
    /// residual tail, and what the decoder holds *for it*. The one field of
    /// the model that spans channels is the output shift, and no decision
    /// consults it.
    ///
    /// Each thread therefore carries its own copy of the three things that
    /// cross units — the sample history, the residual tail, and the channel's
    /// `Held` — and the sequence it replays is exactly the serial one: a
    /// restarting unit clears the channel, states its unpredicted block, then
    /// decides the rest against that; every other unit decides against what
    /// the block before it left. The writer states the same codings again as
    /// it emits them, which is where the model this walk mirrors actually
    /// moves.
    fn decide_interval(&mut self, units: usize, order: Option<usize>) {
        let width = self.coded.frame_size;
        for (channel, plane) in self.interval_planes.iter_mut().enumerate() {
            plane.clear();
            plane.resize(units * width, 0);
            self.interval_decided[channel].clear();
            self.interval_decided[channel].resize(units, Decided::default());
        }

        let prepared = &self.prepared;
        let past = &self.past;
        let past_residuals = &self.past_residuals;
        let modes = &self.second_mode;
        let split_at = RESTART_BLOCK;

        let body = |channel: usize,
                    planes: &mut Vec<i32>,
                    decided: &mut Vec<Decided>,
                    restart_slot: &mut Decided,
                    restated: &mut u64| {
            // The three things that cross units, this channel's own.
            let mut history = past[channel].clone();
            let mut residuals = past_residuals[channel].clone();
            // What the decoder holds for this channel when the interval
            // starts, which is nothing: every interval opens with a restart
            // header, and a restart header throws away what it held.
            let mut held = crate::model::Held::default();

            for index in 0..units {
                let samples = &prepared[channel][index * width..(index + 1) * width];
                let plane = &mut planes[index * width..(index + 1) * width];
                let restart = index == 0;
                let split = if restart { split_at } else { 0 };

                if restart {
                    // A restart header throws away what the decoder was
                    // holding for this channel.
                    held = crate::model::Held::default();
                    let (coding, _) = filter::plain(&samples[..split], &mut plane[..split]);
                    let write_iir = held.restates_iir(&coding.filter);
                    *restart_slot = Decided {
                        // Stated unconditionally: see `decide_restart_block`.
                        write_fir: true,
                        write_iir,
                        write_params: true,
                        coding,
                    };
                    *restated += 1;
                    held.state(&coding);
                }

                let state = if restart {
                    filter::state_from(&samples[..split], &plane[..split])
                } else {
                    filter::state_from(history.window(), &residuals)
                };
                let (coding, _) = filter::choose_fixed(
                    modes[channel],
                    order,
                    history.window(),
                    state,
                    &samples[split..],
                    &mut plane[split..],
                    &if restart {
                        crate::filter::Filter::default()
                    } else {
                        held.in_force()
                    },
                );
                let write_fir = held.restates_fir(&coding.filter);
                let write_iir = held.restates_iir(&coding.filter);
                decided[index] = Decided {
                    write_fir,
                    write_iir,
                    write_params: write_fir || write_iir || held.restates(&coding),
                    coding,
                };
                if write_fir {
                    *restated += 1;
                }
                held.state(&coding);

                // What the decoder will carry: the matrixed samples, and the
                // residuals that were actually written.
                history.push(samples);
                residuals.clear();
                residuals.extend_from_slice(&plane[plane.len().saturating_sub(MAX_IIR_ORDER)..]);
            }
            (history, residuals)
        };

        let mut restated = vec![0u64; self.channels];
        let carried: Vec<(History, Vec<i32>)> = {
            #[cfg(feature = "parallel")]
            if let Some(pool) = &self.threads.0 {
                use rayon::prelude::*;
                let planes = &mut self.interval_planes;
                let decided = &mut self.interval_decided;
                let restarts = &mut self.interval_restart;
                pool.install(|| {
                    planes
                        .par_iter_mut()
                        .zip(decided.par_iter_mut())
                        .zip(restarts.par_iter_mut())
                        .zip(restated.par_iter_mut())
                        .enumerate()
                        .map(|(channel, (((p, d), r), n))| body(channel, p, d, r, n))
                        .collect()
                })
            } else {
                self.interval_planes
                    .iter_mut()
                    .zip(self.interval_decided.iter_mut())
                    .zip(self.interval_restart.iter_mut())
                    .zip(restated.iter_mut())
                    .enumerate()
                    .map(|(channel, (((p, d), r), n))| body(channel, p, d, r, n))
                    .collect()
            }
            #[cfg(not(feature = "parallel"))]
            self.interval_planes
                .iter_mut()
                .zip(self.interval_decided.iter_mut())
                .zip(self.interval_restart.iter_mut())
                .zip(restated.iter_mut())
                .enumerate()
                .map(|(channel, (((p, d), r), n))| body(channel, p, d, r, n))
                .collect()
        };

        for (channel, (history, residuals)) in carried.into_iter().enumerate() {
            self.past[channel] = history;
            self.past_residuals[channel] = residuals;
        }
        self.stats.filters_restated += restated.iter().sum::<u64>();
    }
}

/// What writes the stream.
///
/// Everything that changes as units are written — what the decoder is
/// holding, when each unit arrives, what is due to be restated, what is
/// signed — and the interval being written, under the names the encoder uses
/// for the same things while deciding them. Taken out of the encoder to write
/// one interval while the next is decided: nothing here is read by a
/// decision, and nothing a decision changes is read here except through the
/// [`Record`] it hands over.
#[derive(Debug)]
struct Writer {
    coded: Coded,
    channels: usize,
    /// The second filter the decoder currently holds, per channel. A restart
    /// header clears it, and it is almost always empty — so saying
    /// "unchanged" rather than "order zero" saves four bits a channel an
    /// access unit, which is more than the filter itself was winning.
    ///
    /// The whole taps, not just the order: two filters of the same order with
    /// different coefficients are different filters, and saying "unchanged"
    /// there leaves a decoder predicting with the previous ones. That produced
    /// a stream that decoded to the wrong samples with every checksum intact,
    /// and only the round trip caught it.
    /// The high-resolution output timing field, one bit per restart header —
    /// see [`crate::hires`], where the run-length code and why it is written
    /// at all are set out.
    hires: crate::hires::Timing,
    /// What the decoder holds, as the writer left it: both filters, the
    /// coding, the shifts, the block size and the dynamic range gain. Queried
    /// for every "does this have to be said again?" decision and updated by
    /// the writer as it emits — see [`crate::model`].
    model: crate::model::DecoderModel,
    /// Scratch for costing a rematrixed channel before committing to it.
    input_timing: u16,
    output_timing: u16,
    /// The decoder's input buffer, in samples: how far the arrival clock is
    /// behind the presentation clock. It starts full, a large unit spends from
    /// it, and the units after one refill it. Never negative in a stream a
    /// decoder will accept — see [`Stats::starved`].
    advance: i64,
    /// The lossless check accumulated since the last restart header, per
    /// substream, which is what the *next* one certifies.
    ///
    /// Per substream because each covers every channel up to *its* last one:
    /// the stereo presentation certifies two channels, the one behind it
    /// certifies all of them.
    pending_check: [u32; MAX_SUBSTREAMS],
    /// What to carry in the next access unit's extra data, and under which
    /// Evolution identifier. Held in one buffer that is refilled rather than
    /// reallocated, and emptied once written: object metadata belongs to the
    /// unit it describes.
    evolution: Vec<u8>,
    evolution_id: u32,
    /// The sample of the next unit the payload takes effect at — see
    /// [`Encoder::set_evolution_at`].
    evolution_offset: u32,
    /// Whether there is one to write. Not "is the buffer empty": a payload of
    /// no bytes is a payload, and it is not the same as carrying none.
    evolution_pending: bool,
    /// The key each Evolution frame's protection field is signed with, when
    /// there is one. See [`crate::protection`].
    protection: Option<hz_core::hmac::HmacSha256>,
    /// Access units since each substream last stated one. The value stands for
    /// `2^refresh` of them and no longer, so this is a deadline and not a
    /// preference.
    since_range: [u64; MAX_SUBSTREAMS],
    frames_written: u64,
    /// Per channel, how this unit's last block is coded.
    codings: Vec<Coding>,
    /// And how the unprediced first block of a restarting unit is.
    restart_codings: Vec<Coding>,
    /// Whether each channel's second filter has to be described, per block.
    write_fir: Vec<bool>,
    restart_write_fir: Vec<bool>,
    write_iir: Vec<bool>,
    restart_write_iir: Vec<bool>,
    /// Whether each channel's parameters are said at all in the unit's
    /// blocks, and in the unpredicted first block of a restarting unit. A
    /// channel says nothing when its filters, codebook and width are all what
    /// the decoder already holds, which costs one bit rather than eleven.
    write_params: Vec<bool>,
    restart_write_params: Vec<bool>,
    /// The samples of the unit being encoded, before prediction.
    samples: Vec<Vec<i32>>,
    /// Channel-major planes of residuals, allocated once and reused.
    planes: Vec<Vec<i32>>,
    /// The dead-bit shift in force, per channel, as the units are written.
    output_shift: Vec<u8>,
    /// What the interval being written states for its dynamic range.
    dynamic_range: [Option<DynamicRange>; MAX_SUBSTREAMS],
    /// Units into the restart interval being written.
    since_restart: u64,
    /// What writing counts; see [`Encoder::stats`].
    stats: Stats,
    prepared: Vec<Vec<i32>>,
    prepared_shifts: Vec<u8>,
    prepared_checks: Vec<[u32; MAX_SUBSTREAMS]>,
    interval_decided: Vec<Vec<Decided>>,
    interval_planes: Vec<Vec<i32>>,
    interval_restart: Vec<Decided>,
    matrices: Vec<Primitive>,
    presentations: Vec<Vec<Primitive>>,
    arrangement: Vec<Primitive>,
    assignment: Vec<Vec<u8>>,
    presentation_shift: Vec<Vec<u8>>,
    max_shift: u8,
    max_output_bits: u8,
}

/// An interval decided and not yet written: its units, and everything the
/// deciding left for the writing.
#[derive(Debug)]
struct Record {
    held: Vec<Held>,
    dynamic_range: [Option<DynamicRange>; MAX_SUBSTREAMS],
    prepared: Vec<Vec<i32>>,
    prepared_shifts: Vec<u8>,
    prepared_checks: Vec<[u32; MAX_SUBSTREAMS]>,
    interval_decided: Vec<Vec<Decided>>,
    interval_planes: Vec<Vec<i32>>,
    interval_restart: Vec<Decided>,
    matrices: Vec<Primitive>,
    presentations: Vec<Vec<Primitive>>,
    arrangement: Vec<Primitive>,
    assignment: Vec<Vec<u8>>,
    presentation_shift: Vec<Vec<u8>>,
    max_shift: u8,
    max_output_bits: u8,
}

impl Writer {
    /// Write an interval, and hand back its bytes, its units' buffers and
    /// how long the writing took.
    fn write(&mut self, record: Record) -> (Vec<u8>, Vec<Held>, std::time::Duration) {
        let start = std::time::Instant::now();
        let Record {
            held,
            dynamic_range,
            prepared,
            prepared_shifts,
            prepared_checks,
            interval_decided,
            interval_planes,
            interval_restart,
            matrices,
            presentations,
            arrangement,
            assignment,
            presentation_shift,
            max_shift,
            max_output_bits,
        } = record;
        self.dynamic_range = dynamic_range;
        self.prepared = prepared;
        self.prepared_shifts = prepared_shifts;
        self.prepared_checks = prepared_checks;
        self.interval_decided = interval_decided;
        self.interval_planes = interval_planes;
        self.interval_restart = interval_restart;
        self.matrices = matrices;
        self.presentations = presentations;
        self.arrangement = arrangement;
        self.assignment = assignment;
        self.presentation_shift = presentation_shift;
        self.max_shift = max_shift;
        self.max_output_bits = max_output_bits;

        let mut out = Vec::with_capacity(held.len() * 4096);
        for (index, unit) in held.iter().enumerate() {
            self.evolution_pending = unit.evolution.is_some();
            self.evolution_id = unit.evolution.unwrap_or(0);
            self.evolution_offset = unit.evolution_offset;
            self.evolution.clear();
            self.evolution.extend_from_slice(&unit.payload);
            out.extend_from_slice(&self.encode(index, unit.frames, unit.padding));
        }
        (out, held, start.elapsed())
    }

    /// One access unit, from samples to bytes.
    ///
    /// The steps are named rather than inlined because each of them is a
    /// decision with its own reasons, and the reasons are what the next change
    /// needs. `padding` is how many of `frames` are not real samples, which
    /// only a final short unit has.
    fn encode(&mut self, index: usize, frames: usize, padding: usize) -> Vec<u8> {
        let _ = frames;
        let width = self.coded.frame_size;
        for (channel, plane) in self.samples.iter_mut().enumerate() {
            plane.copy_from_slice(&self.prepared[channel][index * width..(index + 1) * width]);
            self.output_shift[channel] = self.prepared_shifts[index * self.channels + channel];
        }
        let check = self.prepared_checks[index];

        let restart = self.since_restart == 0;
        // A restart header throws away everything the blocks before it left
        // standing: both filters, the coding, the block size and the shifts.
        // The writer does this again where the header is actually written; it
        // is done here too because the decisions below are made against what
        // the decoder will be holding, not against what it holds now.
        if restart {
            for substream in 0..self.coded.substreams {
                self.model.restart(substream, &self.coded.ranges[substream]);
            }
        }
        // A stream whose content fills its container never says anything about
        // shifts at all.
        let write_shift = self
            .model
            .restates_shifts(&self.output_shift, self.channels - 1);

        // A restart header clears the decoder's matrices, so they are said
        // there and nowhere else.
        // A restart header clears the decoder's matrices, so they are said
        // again on every restart — and a presentation is a matrix like any
        // other, so a stream that states one has to say so even when it
        // rematrixes nothing for compression.
        let write_matrix = restart
            && (!self.matrices.is_empty()
                || !self.arrangement.is_empty()
                || self.presentations.iter().any(|rows| !rows.is_empty()));

        // A restarting unit is two blocks: an unpredicted one that refills
        // the decoder's filter state, and the rest of the unit coded from it.
        // See [`RESTART_BLOCK`].
        let split = if restart { RESTART_BLOCK } else { 0 };
        // Already decided, for the whole interval — see `decide_interval`.
        // What is left here is to put this unit's share where the writer looks
        // for it, which is a copy of forty samples a channel.
        for channel in 0..self.channels {
            let decided = &self.interval_decided[channel][index];
            self.codings[channel] = decided.coding;
            self.write_fir[channel] = decided.write_fir;
            self.write_iir[channel] = decided.write_iir;
            self.write_params[channel] = decided.write_params;
            self.planes[channel].copy_from_slice(
                &self.interval_planes[channel][index * width..(index + 1) * width],
            );
            if restart {
                let first = &self.interval_restart[channel];
                self.restart_codings[channel] = first.coding;
                self.restart_write_fir[channel] = first.write_fir;
                self.restart_write_iir[channel] = first.write_iir;
                self.restart_write_params[channel] = first.write_params;
            }
        }

        let certified = self.certify(restart);
        let bytes = self.write_unit(Written {
            restart,
            split,
            write_shift,
            write_matrix,
            certified,
            padding,
        });

        for (accumulated, unit_check) in self.pending_check.iter_mut().zip(&check) {
            *accumulated ^= unit_check;
        }
        self.schedule_arrival(bytes.len());
        self.frames_written += self.coded.frame_size as u64;
        bytes
    }

    /// Build this unit's blocks, write it, and account for what it chose.
    ///
    /// The accounting is here rather than in a function of its own because the
    /// blocks borrow the codings and the statistics are a field beside them;
    /// splitting them would mean either copying the blocks or handing out a
    /// borrow the counter cannot hold.
    fn write_unit(&mut self, written: Written) -> Vec<u8> {
        let Written {
            restart,
            split,
            write_shift,
            write_matrix,
            certified,
            padding,
        } = written;

        // Decided before the blocks, which borrow the codings for as long as
        // they exist.
        let dynamic_range = self.due_dynamic_range();
        // What a decoder that never left is running, per substream: the word
        // it read last, which is what it holds the start-up gain against.
        let running: Vec<Option<DynamicRange>> = (0..self.coded.substreams)
            .map(|index| self.model.range(index))
            .collect();

        // A restart header sets the decoder's block size back to eight, so
        // that is what it holds coming into a restarting unit whatever the
        // last block said.
        let mut held = self.model.blocksize(0);
        let first_frames = if restart {
            split
        } else {
            self.coded.frame_size
        };
        let tail_frames = self.coded.frame_size - split;
        let all = [
            Block {
                first: 0,
                frames: first_frames,
                write_frames: {
                    let differs = first_frames != held;
                    held = first_frames;
                    differs
                },
                codings: if restart {
                    &self.restart_codings
                } else {
                    &self.codings
                },
                write_fir: if restart {
                    &self.restart_write_fir
                } else {
                    &self.write_fir
                },
                write_iir: if restart {
                    &self.restart_write_iir
                } else {
                    &self.write_iir
                },
                write_params: if restart {
                    &self.restart_write_params
                } else {
                    &self.write_params
                },
                output_shift: &self.output_shift,
                write_shift,
            },
            Block {
                first: split,
                frames: tail_frames,
                write_frames: tail_frames != held,
                codings: &self.codings,
                write_fir: &self.write_fir,
                write_iir: &self.write_iir,
                write_params: &self.write_params,
                output_shift: &self.output_shift,
                // A restarting unit says them in its first block; the second
                // has nothing to add.
                write_shift: false,
            },
        ];
        let blocks = &all[..if restart { 2 } else { 1 }];

        let unit = Unit {
            samples: &self.planes,
            blocks,
            matrices: &self.matrices,
            arrangement: &self.arrangement,
            presentation_shift: &self.presentation_shift,
            assignment: &self.assignment,
            presentations: &self.presentations,
            write_matrix,
            input_timing: self.input_timing,
            output_timing: self.output_timing,
            lossless_check: certified,
            // Only a restart header carries one, so only a restarting unit
            // takes a bit off the serialiser.
            hires_timing: restart && self.hires.next(self.frames_written),
            max_shift: self.max_shift,
            max_output_bits: self.max_output_bits,
            shorten_by: padding as u16,
            restart,
            evolution: self.evolution_pending.then_some(frame::Evolution {
                id: self.evolution_id,
                bytes: self.evolution.as_slice(),
                sample_offset: self.evolution_offset,
                config: Default::default(),
            }),
            protection: self.protection.as_ref(),
            dynamic_range,
            // A decoder joining at this unit may not be told to start louder
            // than the gain the stream is already running at, so this follows
            // the gains rather than being chosen.
            start_up_gain: crate::format::start_up_gain(&self.dynamic_range, &running),
        };
        let bytes = frame::access_unit(&self.coded, &unit, &mut self.model);
        self.evolution.clear();
        self.evolution_offset = 0;
        self.evolution_pending = false;
        self.since_restart = (self.since_restart + 1) % RESTART_INTERVAL;

        // The real cost, code lengths included — counting the raw tails alone
        // made the codebooks look free and their gain look like nothing.
        let mut residual_bits = 0u64;
        for block in blocks {
            for (channel, coding) in block.codings.iter().enumerate() {
                self.stats.orders[coding.filter.fir.order] += 1;
                self.stats.filter_blocks += 1;
                if block.write_params[channel] {
                    self.stats.params_restated += 1;
                }
                self.stats.iir_orders[coding.filter.iir.order] += 1;
                self.stats.codebooks[coding.codebook as usize] += 1;
                let residuals = &self.planes[channel][block.first..block.first + block.frames];
                residual_bits +=
                    crate::huffman::cost(coding.codebook, coding.huff_lsbs, 0, residuals)
                        .unwrap_or(0) as u64;
            }
        }
        self.stats.units += 1;
        if !self.matrices.is_empty() {
            self.stats.matrixed_units += 1;
        }
        self.stats.residual_bits += residual_bits;
        self.stats.residuals += self.channels as u64 * self.coded.frame_size as u64;
        self.stats.total_bytes += bytes.len() as u64;
        self.stats.peak_unit_bytes = self.stats.peak_unit_bytes.max(bytes.len() as u64);
        self.stats.framing_bytes += bytes.len() as u64 - residual_bits.div_ceil(8);
        bytes
    }

    /// Read and clear the check a restart header certifies.
    ///
    /// A restart header certifies everything since the previous one, so the
    /// accumulator is read and cleared here and this unit's own check goes
    /// into the next interval.
    fn certify(&mut self, restart: bool) -> [u8; MAX_SUBSTREAMS] {
        let mut certified = [0u8; MAX_SUBSTREAMS];
        if restart {
            for (slot, accumulated) in certified.iter_mut().zip(&mut self.pending_check) {
                *slot = xor_to_byte(*accumulated);
                *accumulated = 0;
            }
        }
        certified
    }

    /// When the next access unit is declared to arrive.
    ///
    /// Long enough for the one just written to have got there at the declared
    /// peak, and otherwise whatever steers the buffer back to full — which is
    /// one frame exactly once nothing has been borrowed.
    fn schedule_arrival(&mut self, bytes: usize) {
        let frames = self.coded.frame_size as i64;
        let reserve = (self.coded.frame_size * INPUT_RESERVE) as i64;
        let peak = u64::from(self.coded.peak_bitrate).max(1);
        // From the unit just written, which is the one whose bytes have to
        // reach a decoder before the next one is due — not the one before it.
        let words = (bytes / 2) as u64;
        let needed = (words << 8).div_ceil(peak);
        let interval = (needed as i64).max(self.advance + frames - reserve).max(1);

        self.advance += frames - interval;
        if self.advance < 0 {
            // The content needs more than the stream declared, on average and
            // not merely in one unit — no schedule fixes that, and a decoder
            // will report the units as arriving faster than it can take them.
            self.stats.starved += 1;
        }
        self.stats.lowest_advance = self.stats.lowest_advance.min(self.advance);

        self.input_timing = self.input_timing.wrapping_add(interval as u16);
        self.output_timing = self
            .output_timing
            .wrapping_add(self.coded.frame_size as u16);
    }

    /// The directory's second word, per substream: said when the value
    /// changes, and again whenever the last one is about to lapse.
    fn due_dynamic_range(&mut self) -> [Option<DynamicRange>; MAX_SUBSTREAMS] {
        let mut dynamic_range = [None; MAX_SUBSTREAMS];
        for (index, slot) in dynamic_range
            .iter_mut()
            .enumerate()
            .take(self.coded.substreams)
        {
            let Some(wanted) = self.dynamic_range[index] else {
                continue;
            };
            // Counted first and tested after, because a decoder counts this
            // access unit before it looks at the word in it: the value has to
            // arrive on the `2^refresh`-th unit, not the one after. Testing
            // before incrementing puts every word one unit late and a decoder
            // says so, once per substream per interval.
            self.since_range[index] += 1;
            let due = self.model.range(index) != Some(wanted)
                || self.since_range[index] >= wanted.deadline();
            if due {
                *slot = Some(wanted);
                self.since_range[index] = 0;
            }
        }
        dynamic_range
    }
}

/// How far a presentation's rows, as written, land from what it asks for: the
/// largest error over its outputs, each against the size of its own row, in
/// the elements' coordinates.
///
/// The rows are rounded to the field's step and each output is scaled up by
/// its channel's shift afterwards, so a coefficient's rounding comes out
/// `2^shift` times larger — on a source channel at full size, a large error
/// from a small step.
fn presentation_error(
    held: &[Vec<f64>],
    presentation: &crate::hierarchy::Presentation,
    rows: &[Primitive],
    order: &[u8],
    shifts: &[u8],
) -> f64 {
    let n = presentation.channels.min(held.len());
    let decoded = crate::hierarchy::as_decoded(held, rows, order, n);
    let mut worst = 0.0f64;
    for (output, wanted) in presentation.rows.iter().enumerate().take(n) {
        let Some(channel) = order.iter().position(|o| usize::from(*o) == output) else {
            return f64::INFINITY;
        };
        let up = f64::from(shifts.get(channel).copied().unwrap_or(0)).exp2();
        let size = wanted.iter().map(|w| w * w).sum::<f64>().sqrt();
        if size == 0.0 {
            continue;
        }
        let error = decoded[output]
            .iter()
            .zip(wanted)
            .map(|(got, want)| (got * up - want).powi(2))
            .sum::<f64>()
            .sqrt();
        worst = worst.max(error / size);
    }
    worst
}

/// How far a presentation's rows may land from its fold, as a fraction of
/// each output's own row — see [`presentation_error`].
///
/// Three in ten thousand. On the programme slice it puts the 5.1 and 7.1
/// within about −108 dBFS of the fold, which is what the reference's shifts of
/// three to five give, for 2.7 % of the stream from the decorrelation it
/// allows. A thousandth gives 3.4 % at −101 dBFS; a ten-thousandth, 0.6 % at
/// −115; three in a hundred thousand is tighter than the rows one shift for
/// every channel writes, and no interval keeps its folds.
const PRESENTATION_PRECISION: f64 = 3e-4;

/// Full scale in the samples a decoder hands back.
const FULL_SCALE: f64 = 8_388_608.0;

/// What a presentation's rows may land off its fold by, in samples, as the
/// room a shift has to leave under full scale on top of its own rounding:
/// [`PRESENTATION_PRECISION`] of full scale, eight times over for a bound
/// that holds at a peak rather than on average.
const FOLD_MARGIN: f64 = 8.0 * PRESENTATION_PRECISION * FULL_SCALE;

/// Rows, output order and stated shifts of every early presentation.
type PresentationRows = (Vec<Vec<Primitive>>, Vec<Vec<u8>>, Vec<Vec<u8>>);

/// The most steps the last substream's cascade may take once it decorrelates,
/// which leaves four of its sixteen for coding matrices.
const DECORRELATING_STEPS_AT_MOST: usize = 12;

/// What a decorrelating step has to take out of its channel over an interval,
/// as the fit models it, in bits. See [`Encoder::decorrelate`] for what it is
/// and is not.
const DECORRELATION_BITS: f64 = 5_000.0;

/// The weights that predict channel `k` from channels `0..k` by least squares,
/// from their Gram matrix; `None` if the channels are too nearly dependent to
/// say.
fn least_squares(gram: &[Vec<f64>], k: usize) -> Option<Vec<f64>> {
    let scale = (0..k).map(|i| gram[i][i]).fold(0.0f64, f64::max);
    if scale <= 0.0 {
        return None;
    }
    // The normal equations, with the right-hand side as a last column and a
    // touch of ridge so that a silent channel does not make them singular.
    let mut a: Vec<Vec<f64>> = (0..k)
        .map(|i| {
            let mut row = gram[i][..=k].to_vec();
            row[i] += scale * 1e-9;
            row
        })
        .collect();
    for col in 0..k {
        let pivot = (col..k).max_by(|x, y| a[*x][col].abs().total_cmp(&a[*y][col].abs()))?;
        if a[pivot][col].abs() < scale * 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        for row in col + 1..k {
            let factor = a[row][col] / a[col][col];
            for c in col..=k {
                a[row][c] -= factor * a[col][c];
            }
        }
    }
    let mut w = vec![0.0; k];
    for i in (0..k).rev() {
        let sum: f64 = (i + 1..k).map(|j| a[i][j] * w[j]).sum();
        w[i] = (a[i][k] - sum) / a[i][i];
    }
    Some(w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::SampleBits;

    /// A channel holding a hundredth of its element can only be written at a
    /// shift of six, which rounds its output to sixty-four samples. Where its
    /// fold peaks at half of full scale that is taken; where it peaks at full
    /// scale it would round past it and wrap, so a decorrelating step asking
    /// for it is refused, and the rows the stream would otherwise write fall
    /// back to one shift for every channel.
    #[test]
    fn a_shift_leaves_a_loud_fold_room_to_round() {
        let held = vec![vec![1.0, 0.0], vec![0.0, 0.01]];
        let identity = |channels: usize| crate::hierarchy::Presentation {
            channels,
            rows: (0..channels)
                .map(|channel| {
                    let mut row = vec![0.0; 2];
                    row[channel] = 1.0;
                    row
                })
                .collect(),
        };
        let internal = [identity(2), identity(2)];
        let threads = Threads::new(2);

        let quiet = [vec![0.0, 0.5 * FULL_SCALE]];
        let (_, _, stated) =
            Encoder::presentation_rows(&threads, &held, &internal, 2, false, true, &quiet)
                .expect("rows");
        assert_eq!(stated[0], [0, 6]);

        let loud = [vec![0.0, 0.9999 * FULL_SCALE]];
        assert!(
            Encoder::presentation_rows(&threads, &held, &internal, 2, false, true, &loud).is_none(),
            "a step is refused rather than written to wrap"
        );
        let (_, _, stated) =
            Encoder::presentation_rows(&threads, &held, &internal, 2, false, false, &loud)
                .expect("rows");
        assert!(
            stated[0].windows(2).all(|pair| pair[0] == pair[1]),
            "without the step, one shift for every channel as before: {:?}",
            stated[0]
        );
    }

    #[test]
    fn the_check_folds_a_word_into_a_byte() {
        assert_eq!(xor_to_byte(0x0000_00ab), 0xab);
        assert_eq!(xor_to_byte(0x00ab_0000), 0xab);
        assert_eq!(xor_to_byte(0xabab_abab), 0x00);
    }

    #[test]
    fn a_short_last_unit_is_marked_as_shortened() {
        let [marker, shorten] = end_of_stream(7);
        assert_eq!(marker, 0xd234);
        assert_eq!(shorten & 0x1fff, 7);
        assert_ne!(shorten & 0x2000, 0, "a decoder needs this bit to shorten");
    }

    /// A stream that ends on a unit boundary must not gain an empty one:
    /// a decoder reports it as a unit with nothing in it, which looks like a
    /// truncated stream.
    #[test]
    fn a_stream_that_ends_on_a_boundary_gains_no_empty_unit() {
        let mut encoder = Encoder::new(Config {
            sample_rate: 48_000,
            channels: 2,
            bits: SampleBits::TwentyFour,
        })
        .unwrap();
        let unit = vec![0i32; encoder.frame_size() * 2];
        // The interval is held, so a push before it fills writes nothing; the
        // flush is where the bytes come from.
        assert!(encoder.push(&unit).is_empty());
        let stream = encoder.finish(&[]);
        let units = crate::reader::stream(&stream).expect("our own stream reads");
        assert_eq!(units.len(), 1, "one unit in, one unit out");
        assert!(encoder.finish(&[]).is_empty(), "and nothing after it");
    }

    /// The channel counts the format's arrangements name are the ones this
    /// writes; anything else is refused by name rather than approximated.
    /// Shipped streams carry twelve, fourteen and sixteen elements, and the
    /// only thing that differs between them is where the fourth substream
    /// ends. Everything else about the description is identical, so the sizes
    /// between are written too rather than being singled out.
    #[test]
    fn every_element_count_an_object_programme_can_have_is_written() {
        for channels in 9..=crate::format::MAX_CHANNELS {
            let encoder = Encoder::new(Config {
                sample_rate: 48_000,
                channels,
                bits: SampleBits::TwentyFour,
            })
            .unwrap_or_else(|e| panic!("{channels} channels should encode: {e}"));
            let presentations = encoder.presentations();
            assert_eq!(
                presentations,
                [2, 6, 8, channels],
                "{channels} channels: a decoder stopping early gets 2, 6 and 8"
            );
        }
    }

    #[test]
    fn a_channel_count_with_no_arrangement_is_refused_by_name() {
        // Nine and up are object programmes; below that only the layouts the
        // arrangement fields can name.
        for channels in [3usize, 5, 7, 17, 24] {
            let error = Encoder::new(Config {
                sample_rate: 48_000,
                channels,
                bits: SampleBits::TwentyFour,
            })
            .unwrap_err();
            assert!(
                error.to_string().contains(&format!("{channels} channels")),
                "{error}"
            );
        }
    }

    #[test]
    fn six_and_eight_channels_are_written() {
        for channels in [6usize, 8] {
            assert!(
                Encoder::new(Config {
                    sample_rate: 48_000,
                    channels,
                    bits: SampleBits::TwentyFour,
                })
                .is_ok(),
                "{channels} channels should encode"
            );
        }
    }
}

/// How many low bits every sample here has to spare.
///
/// `None` for a block of pure silence, which is divisible by everything and so
/// asks for the largest shift going — and would hand it straight back at the
/// next block with any content in it. The caller holds what it already had
/// instead, which keeps the coded domain still across a quiet passage.
fn wasted_bits(samples: &[i32]) -> Option<u8> {
    let mut bits = 0u32;
    for sample in samples {
        if *sample != 0 {
            bits |= sample.unsigned_abs();
        }
    }
    if bits == 0 {
        return None;
    }
    Some((bits.trailing_zeros() as u8).min(crate::frame::MAX_OUTPUT_SHIFT))
}

#[cfg(test)]
mod shift_tests {
    use super::*;

    #[test]
    fn content_that_fills_its_container_has_nothing_to_spare() {
        assert_eq!(wasted_bits(&[1, 2, 3]), Some(0));
        assert_eq!(wasted_bits(&[-1]), Some(0));
        assert_eq!(wasted_bits(&[i32::MIN >> 1, 3]), Some(0));
    }

    #[test]
    fn twenty_bits_inside_twenty_four_gives_four_back() {
        let samples: Vec<i32> = (1..40).map(|v| v << 4).collect();
        assert_eq!(wasted_bits(&samples), Some(4));
        // One sample that uses the low bits is enough to spend them all.
        let mut spoiled = samples.clone();
        spoiled[7] |= 1;
        assert_eq!(wasted_bits(&spoiled), Some(0));
    }

    #[test]
    fn negative_samples_count_their_own_low_bits() {
        // Taking the sign bit for a set low bit would report nothing to spare
        // on any block with a negative sample in it, which is every block.
        let samples: Vec<i32> = (1..40).map(|v| -(v << 3)).collect();
        assert_eq!(wasted_bits(&samples), Some(3));
    }

    #[test]
    fn silence_asks_for_nothing_rather_than_for_everything() {
        // Zero is divisible by every power of two, so a silent block would
        // claim the largest shift there is and hand it back at the next block
        // with content. The caller holds what it had instead.
        assert_eq!(wasted_bits(&[0, 0, 0]), None);
        assert_eq!(wasted_bits(&[]), None);
    }

    #[test]
    fn a_shift_is_never_larger_than_the_field_can_say() {
        // The field is four bits and signed, so seven.
        let samples: Vec<i32> = (1..8).map(|v| v << 20).collect();
        assert_eq!(wasted_bits(&samples), Some(crate::frame::MAX_OUTPUT_SHIFT));
    }
}
