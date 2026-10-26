//! Taking the extension waveforms out of the bed they are folded into.
//!
//! A DTS:X lossless bed is the speakers with every extension waveform already
//! mixed in, at gains the stream states (or, for a few streams, at gains the
//! encoder's own panner chose). The sixteen-element stream carries the same
//! waveforms again as elements of their own, so an object renderer would play
//! each of them twice: once from the bed speaker it was folded into and once
//! from its element. This cleans the bed first, the way the decoder does when
//! it renders the waveforms on their own.
//!
//! Per frame:
//!
//! 1. the folds the metadata states (`FoldPlan::from_metadata`); a stated Q15
//!    fold is taken out with the encoder's own rounded multiply, so the bed
//!    comes back bit for bit as it was before the fold;
//! 2. the folds it does not state: the ones the encoder panned from the
//!    object's position are recomputed (`FoldRenderer`), the others fitted to
//!    the audio (`FoldEstimator`). These are not bit exact, and never add
//!    energy to a speaker;
//! 3. only the waveforms the stream carries are taken out. A waveform whose
//!    fold is still unknown is not carried (its element is described as not
//!    there and its plane is silent) and stays in the bed; so does the
//!    waveform dropped when there are nine and the elements hold eight
//!    ([`dropped`]).
//!
//! Every plane the encoder is given is therefore "the bed without the carried
//! waveforms" or "one carried waveform", and the narrow presentations are
//! cut from the cleaned bed.
//!
//! The fold state is allocated once and reused: nothing is allocated for a
//! frame once the first has been seen.

use dca::{BedFold, FoldEstimator, FoldPlan, FoldRenderer, MAX_SOURCES, SourceRole, XMetadata};

use crate::{BED_SPEAKER_MAP, CHANNELS};

/// Extension waveforms the stream has elements for.
pub(crate) const CARRIED: usize = 8;
/// Speakers the fold machinery indexes (DCA speaker numbers).
const SPEAKERS: usize = 16;
/// Integer sample to the float the fold machinery works in, and back.
const SCALE: f32 = 8_388_608.0;
/// A fold that puts a waveform into one bed speaker and nowhere else counts
/// as unity within this.
const UNITY_WITHIN: f32 = 0.01;

/// The waveform behind each extension element, in element order; `None` where
/// the element carries nothing (no waveform, or one that cannot be taken out
/// of the bed).
pub(crate) type Slots = [Option<usize>; CARRIED];

/// The waveform to leave in the bed when the frame has more than the
/// elements hold (nine, in the widest profile).
///
/// A waveform already fully represented by its fold costs nothing to leave
/// in the bed: an object at a speaker's own position, folded at unity into
/// that speaker alone, plays from exactly the place its element would put it.
/// The one nearest the front goes first (the centre one, where there is one),
/// the first of them on a tie. With none, the last waveform goes, a height
/// folded across two speakers.
///
/// This is the choice for one frame; [`Cleaner`] makes it once and keeps it,
/// since the elements after the one left out would otherwise change waveform
/// whenever an object moves and its fold stops being unity.
pub(crate) fn dropped(meta: &XMetadata) -> Option<usize> {
    let count = meta.source_count().min(MAX_SOURCES);
    if count <= CARRIED {
        return None;
    }
    let mut best: Option<(usize, f64)> = None;
    for feed in 0..count {
        let Some(source) = meta.source(feed) else {
            continue;
        };
        let (SourceRole::Object { position, .. }, BedFold::Known(columns)) =
            (source.role, source.fold)
        else {
            continue;
        };
        let alone = columns.iter().filter(|g| **g != 0.0).count() == 1
            && columns.iter().any(|g| (*g - 1.0).abs() < UNITY_WITHIN);
        let away = position.azimuth_degrees().abs();
        if alone && best.is_none_or(|(_, so_far)| away < so_far) {
            best = Some((feed, away));
        }
    }
    Some(best.map_or(count - 1, |(feed, _)| feed))
}

/// Which waveform each extension element carries: the frame's waveforms in
/// order less the dropped one, the first eight, with an unknown fold leaving
/// its element empty.
pub(crate) fn slots(meta: &XMetadata, plan: &FoldPlan, drop: Option<usize>) -> Slots {
    let count = meta.source_count().min(MAX_SOURCES);
    let mut slots = [None; CARRIED];
    for (slot, feed) in slots
        .iter_mut()
        .zip((0..count).filter(|f| Some(*f) != drop))
    {
        *slot = plan.source_is_known(feed).then_some(feed);
    }
    slots
}

/// A float sample at the planes' scale back to the integer it was, clamped
/// to the 24 bits the lossless layer holds.
fn to_int(sample: f32) -> i32 {
    ((sample * SCALE).round() as i32).clamp(-(1 << 23), (1 << 23) - 1)
}

/// Fold state and scratch, reused from frame to frame.
pub(crate) struct Cleaner {
    renderer: FoldRenderer,
    estimator: FoldEstimator,
    /// The bed by DCA speaker, as floats; `None` for the speakers that are
    /// not fed to the fit (the LFE and the ones a bed does not have).
    bed: Vec<Option<Vec<f32>>>,
    /// The extension waveforms, as floats.
    sources: Vec<Vec<f32>>,
    /// One cleaned bed channel.
    out: Vec<f32>,
    /// The waveform left in the bed, once chosen: the same one for as long as
    /// the frames have more waveforms than the elements hold.
    left_in_bed: Option<usize>,
}

impl Cleaner {
    pub(crate) fn new() -> Self {
        Self {
            renderer: FoldRenderer::new(),
            estimator: FoldEstimator::new(),
            bed: (0..SPEAKERS).map(|_| None).collect(),
            sources: (0..MAX_SOURCES).map(|_| Vec::new()).collect(),
            out: Vec::new(),
            left_in_bed: None,
        }
    }

    /// The waveform to leave in the bed for this frame: the one chosen
    /// before while the frames still have too many, else the frame's choice.
    fn leave_in_bed(&mut self, meta: &XMetadata) -> Option<usize> {
        let count = meta.source_count().min(MAX_SOURCES);
        if count <= CARRIED {
            self.left_in_bed = None;
        } else if self.left_in_bed.is_none_or(|feed| feed >= count) {
            self.left_in_bed = dropped(meta);
        }
        self.left_in_bed
    }

    /// Forget what the fits learnt (a new stream).
    pub(crate) fn reset(&mut self) {
        self.renderer.reset();
        self.estimator.reset();
        self.left_in_bed = None;
    }

    /// Lay the sixteen planes of a frame of `n` samples into `planes`: the
    /// bed cleaned of the waveforms the frame carries, then those waveforms,
    /// one per element. `bed` is in element order, `ext` in waveform order.
    /// Returns what each extension element carries.
    ///
    /// With no metadata nothing is known of what is folded where: the bed is
    /// left as it is and no waveform is carried.
    pub(crate) fn clean(
        &mut self,
        meta: Option<&XMetadata>,
        bed: &[&[i32]; 8],
        ext: &[&[i32]; MAX_SOURCES],
        n: usize,
        planes: &mut Vec<i32>,
    ) -> Slots {
        planes.clear();
        planes.resize(CHANNELS * n, 0);
        let Some(meta) = meta.filter(|m| m.source_count() > 0) else {
            copy_bed(bed, n, planes);
            return [None; CARRIED];
        };
        let count = meta.source_count().min(MAX_SOURCES);
        self.fill(bed, ext, n, count);
        let mut plan = FoldPlan::from_metadata(meta);
        self.renderer.apply(&mut plan, meta, n);
        if plan.has_unknown() {
            self.estimator.refine(&mut plan, &self.bed, &self.sources);
        }
        self.apply(meta, &plan, bed, ext, n, planes)
    }

    /// The same with the plan given, so that what it leaves unknown can be
    /// tested.
    fn apply(
        &mut self,
        meta: &XMetadata,
        plan: &FoldPlan,
        bed: &[&[i32]; 8],
        ext: &[&[i32]; MAX_SOURCES],
        n: usize,
        planes: &mut [i32],
    ) -> Slots {
        let slots = slots(meta, plan, self.leave_in_bed(meta));
        if n == 0 {
            return slots;
        }
        // Only the carried waveforms are taken out: an empty one is left.
        for feed in 0..MAX_SOURCES {
            if !slots.contains(&Some(feed)) {
                self.sources[feed].clear();
            }
        }
        for (c, plane) in planes.chunks_mut(n).take(8).enumerate() {
            let speaker = BED_SPEAKER_MAP[c];
            let source = bed[c];
            let len = source.len().min(n);
            if c == 0 || !plan.touches(speaker) {
                plane[..len].copy_from_slice(&source[..len]);
                continue;
            }
            let Some(float) = self.bed[speaker].as_deref() else {
                plane[..len].copy_from_slice(&source[..len]);
                continue;
            };
            let float = &float[..len];
            self.out.resize(len, 0.0);
            plan.clean_channel(speaker, float, &self.sources, &mut self.out);
            for (value, cleaned) in plane[..len].iter_mut().zip(&self.out) {
                *value = to_int(*cleaned);
            }
        }
        for (element, feed) in slots.iter().enumerate() {
            if let Some(feed) = *feed {
                let source = ext[feed];
                let len = source.len().min(n);
                let at = (8 + element) * n;
                planes[at..at + len].copy_from_slice(&source[..len]);
            }
        }
        slots
    }

    /// The frame's samples as floats, for the fits and the subtraction.
    fn fill(&mut self, bed: &[&[i32]; 8], ext: &[&[i32]; MAX_SOURCES], n: usize, count: usize) {
        for (c, source) in bed.iter().enumerate().skip(1) {
            let speaker = BED_SPEAKER_MAP[c];
            let len = source.len().min(n);
            let float = self.bed[speaker].get_or_insert_with(Vec::new);
            float.clear();
            float.extend(source[..len].iter().map(|v| *v as f32 / SCALE));
            // A channel the frame lacks is silent, not absent.
            float.resize(n, 0.0);
        }
        for (feed, source) in ext.iter().enumerate() {
            let float = &mut self.sources[feed];
            float.clear();
            if feed < count {
                let len = source.len().min(n);
                float.extend(source[..len].iter().map(|v| *v as f32 / SCALE));
                float.resize(n, 0.0);
            }
        }
    }
}

/// The bed as it is, into the first eight planes.
fn copy_bed(bed: &[&[i32]; 8], n: usize, planes: &mut [i32]) {
    for (c, source) in bed.iter().enumerate() {
        let len = source.len().min(n);
        planes[c * n..c * n + len].copy_from_slice(&source[..len]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dca::{SourceMetadata, SphericalPosition};

    /// The Q15 gain of 1/2, the fold the tests state.
    const HALF: f32 = 0.5;

    fn object(azimuth_degrees: i16, fold: BedFold) -> SourceMetadata {
        SourceMetadata {
            role: SourceRole::Object {
                position: SphericalPosition {
                    azimuth_half_degrees: azimuth_degrees * 2,
                    elevation_half_degrees: 0,
                    distance_64ths: 64,
                },
                centre_height_alternative: false,
            },
            fold,
        }
    }

    /// Columns of the 7.1 reference: C, L, R, LFE, Lsr, Rsr, Lss, Rss.
    fn columns(c: f32, l: f32, r: f32) -> BedFold {
        BedFold::Known([c, l, r, 0.0, 0.0, 0.0, 0.0, 0.0])
    }

    /// What the encoder does to a bed channel: add `gain` of the waveform,
    /// rounded the way the lossless decoder rounds a Q15 multiply.
    fn folded(bed: &[i32], wave: &[i32], gain_q15: i64) -> Vec<i32> {
        bed.iter()
            .zip(wave)
            .map(|(b, w)| b + ((i64::from(*w) * gain_q15 + (1 << 14)) >> 15) as i32)
            .collect()
    }

    fn noise(seed: u32, n: usize, peak: i32) -> Vec<i32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as i32 % (2 * peak + 1)) - peak
            })
            .collect()
    }

    const N: usize = 512;

    /// A frame with a bed of noise and one waveform folded into L and R at
    /// one half, as the lossless layer holds it.
    struct Frame {
        clean_bed: Vec<Vec<i32>>,
        folded_bed: Vec<Vec<i32>>,
        wave: Vec<i32>,
    }

    fn frame() -> Frame {
        let clean_bed: Vec<Vec<i32>> = (0..8).map(|c| noise(10 + c, N, 1 << 20)).collect();
        let wave = noise(99, N, 1 << 21);
        let mut folded_bed = clean_bed.clone();
        // Elements 1 and 2 are L and R.
        folded_bed[1] = folded(&clean_bed[1], &wave, 1 << 14);
        folded_bed[2] = folded(&clean_bed[2], &wave, 1 << 14);
        Frame {
            clean_bed,
            folded_bed,
            wave,
        }
    }

    fn slices(planes: &[Vec<i32>]) -> [&[i32]; 8] {
        std::array::from_fn(|c| planes[c].as_slice())
    }

    fn run(
        cleaner: &mut Cleaner,
        meta: &XMetadata,
        frame: &Frame,
        plan: Option<&FoldPlan>,
    ) -> (Vec<i32>, Slots) {
        let mut ext: [&[i32]; MAX_SOURCES] = [&[]; MAX_SOURCES];
        ext[0] = &frame.wave;
        let bed = slices(&frame.folded_bed);
        let mut planes = Vec::new();
        let slots = match plan {
            None => cleaner.clean(Some(meta), &bed, &ext, N, &mut planes),
            Some(plan) => {
                planes.resize(CHANNELS * N, 0);
                cleaner.fill(&bed, &ext, N, meta.source_count());
                cleaner.apply(meta, plan, &bed, &ext, N, &mut planes)
            }
        };
        (planes, slots)
    }

    /// A stated fold of an object into L and R comes out of the bed bit for
    /// bit, and the object is its own element.
    #[test]
    fn a_stated_fold_leaves_the_bed_exactly() {
        let frame = frame();
        let meta =
            XMetadata::from_sources(&[object(0, columns(0.0, HALF, HALF))]).expect("one source");
        let (planes, slots) = run(&mut Cleaner::new(), &meta, &frame, None);
        assert_eq!(slots[0], Some(0));
        assert!(slots[1..].iter().all(Option::is_none));
        for c in 0..8 {
            assert_eq!(planes[c * N..(c + 1) * N], frame.clean_bed[c], "bed {c}");
        }
        assert_eq!(planes[8 * N..9 * N], frame.wave);
        // The bed really did carry it before.
        assert_ne!(frame.folded_bed[1], frame.clean_bed[1]);
        assert!(planes[9 * N..].iter().all(|v| *v == 0));
    }

    /// A fold nothing can recover leaves the bed untouched, and the element
    /// is not there (and silent).
    #[test]
    fn an_unknown_fold_leaves_the_bed_and_the_element_empty() {
        let frame = frame();
        let meta = XMetadata::from_sources(&[object(0, BedFold::Unknown)]).expect("one source");
        let plan = FoldPlan::from_metadata(&meta);
        assert!(!plan.source_is_known(0));
        let (planes, slots) = run(&mut Cleaner::new(), &meta, &frame, Some(&plan));
        assert!(slots.iter().all(Option::is_none));
        for c in 0..8 {
            assert_eq!(planes[c * N..(c + 1) * N], frame.folded_bed[c], "bed {c}");
        }
        assert!(planes[8 * N..].iter().all(|v| *v == 0));
    }

    /// With no metadata at all the bed is passed through and nothing rides
    /// the extension elements.
    #[test]
    fn no_metadata_means_no_cleaning_and_no_objects() {
        let frame = frame();
        let mut ext: [&[i32]; MAX_SOURCES] = [&[]; MAX_SOURCES];
        ext[0] = &frame.wave;
        let mut planes = Vec::new();
        let slots = Cleaner::new().clean(None, &slices(&frame.folded_bed), &ext, N, &mut planes);
        assert!(slots.iter().all(Option::is_none));
        for c in 0..8 {
            assert_eq!(planes[c * N..(c + 1) * N], frame.folded_bed[c]);
        }
        assert!(planes[8 * N..].iter().all(|v| *v == 0));
    }

    /// Nine waveforms, eight elements: the ninth is never taken out of the
    /// bed, and the one dropped is a speaker object whose fold is the whole
    /// of it.
    #[test]
    fn a_speaker_object_is_the_one_left_in_the_bed() {
        let heights = (0..4).map(|i| {
            let mut columns = [0.0; 8];
            columns[1 + i % 2] = 0.707;
            SourceMetadata {
                role: SourceRole::Height(dca::SpatialChannel::TopFrontLeft),
                fold: BedFold::Known(columns),
            }
        });
        let mut sources = vec![
            object(-30, columns(0.0, 1.0, 0.0)),
            object(30, columns(0.0, 0.0, 1.0)),
            object(0, columns(1.0, 0.0, 0.0)),
            object(-90, BedFold::Unknown),
            object(90, BedFold::Unknown),
        ];
        sources.extend(heights);
        let meta = XMetadata::from_sources(&sources).expect("nine");
        // The centre one, the third waveform.
        assert_eq!(dropped(&meta), Some(2));
        let plan = FoldPlan::from_metadata(&meta);
        let slots = slots(&meta, &plan, dropped(&meta));
        // Eight elements: all but the centre, the unknown ones empty.
        assert_eq!(
            slots,
            [
                Some(0),
                Some(1),
                None,
                None,
                Some(5),
                Some(6),
                Some(7),
                Some(8)
            ]
        );

        // Without a speaker object, the last waveform goes.
        let sources: Vec<_> = (0..9).map(|i| object(10 * i, BedFold::Unknown)).collect();
        let meta = XMetadata::from_sources(&sources).expect("nine");
        assert_eq!(dropped(&meta), Some(8));
        // Eight or fewer: none.
        let meta = XMetadata::from_sources(&sources[..8]).expect("eight");
        assert_eq!(dropped(&meta), None);
    }

    /// The waveform left in the bed stays the same one while an object moves
    /// and its fold stops being the whole of it, and is chosen afresh after a
    /// reset.
    #[test]
    fn the_waveform_left_in_the_bed_does_not_change_with_the_frames() {
        let nine = |centre: BedFold| -> XMetadata {
            let mut sources = vec![
                object(-30, columns(0.0, 1.0, 0.0)),
                object(30, columns(0.0, 0.0, 1.0)),
                object(0, centre),
            ];
            sources.extend((0..6).map(|i| object(40 + 10 * i, BedFold::Unknown)));
            XMetadata::from_sources(&sources).expect("nine")
        };
        let mut cleaner = Cleaner::new();
        let at_rest = nine(columns(1.0, 0.0, 0.0));
        assert_eq!(cleaner.leave_in_bed(&at_rest), Some(2));
        // The centre object moves: its fold is two columns now, and no
        // other waveform is nearer the front. It stays the one.
        let moving = nine(columns(0.9, 0.3, 0.0));
        assert_eq!(dropped(&moving), Some(0));
        assert_eq!(cleaner.leave_in_bed(&moving), Some(2));
        // Fewer waveforms: nothing left in the bed.
        let eight = XMetadata::from_sources(&[object(0, columns(1.0, 0.0, 0.0)); 8]).unwrap();
        assert_eq!(cleaner.leave_in_bed(&eight), None);
        // After a reset the frame decides again.
        cleaner.leave_in_bed(&at_rest);
        cleaner.reset();
        assert_eq!(cleaner.leave_in_bed(&moving), Some(0));
    }
}
