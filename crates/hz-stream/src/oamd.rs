//! The object metadata a DTS:X frame states, as the TrueHD stream carries it.
//!
//! One block of object audio metadata describes the sixteen elements the
//! encoder is given, one for one and in the same order as the planes
//! (`Cleaner::clean` in `bed.rs`):
//!
//! | element | plane | described as |
//! |---|---|---|
//! | 0 | low frequency channel | the LFE (the syntax puts it first) |
//! | 1 to 7 | L, R, C, Ls, Rs, Lrs, Rrs | bed channels, parked at their speaker |
//! | 8 to 15 | extension waveforms | objects at their stated positions, or heights; not there where no waveform is carried |
//!
//! Every frame gets the bed description, so a decoder never has to hold the
//! last one: a frame with no extension, or whose metadata does not parse,
//! describes the bed alone (elements 8 to 15 not there). Both DTS:X profiles
//! the decoder tells apart are described: the standard one (`x_present`:
//! four height waveforms, no object records) and the alternate ones
//! (`x_imax`: object records with a position each, then the four fixed
//! heights, five to nine waveforms in all). Which waveform each element
//! carries is `bed::Slots`: the frame's waveforms in order, less the one left
//! in the bed when there are more than eight, and less any whose fold is
//! unknown.
//!
//! # Where an object is put
//!
//! The stream states a position as an azimuth (0 = front, negative = left),
//! an elevation and a distance, with the speakers of its reference layout at
//! azimuths 0, +-30, +-90 and +-150 degrees on the horizontal ring. The
//! master's room is a cube with the speakers on its floor: the corners (L, R,
//! Lrs, Rrs), the middle of the walls (C, Ls, Rs). A plain spherical
//! conversion puts an object at the left speaker's azimuth half way between L
//! and C, which is not where the stream's own fold puts it, so the mapping is
//! anchored on the speakers instead ([`position_of`]):
//!
//! - **azimuth**: piecewise linear along the cube's floor perimeter between
//!   the anchors -180 (rear centre), -150 (Lrs corner), -90 (Ls), -30 (L
//!   corner), 0 (C), 30, 90, 150 and 180, so a speaker's azimuth lands
//!   exactly on its own element's point and the points between lie on the
//!   wall between;
//! - **elevation**: z is the elevation over 45 degrees, the height layer, up
//!   to z = 1 (the fixed heights are the corners at z = 1, which is where a
//!   45 degree object lands above the same azimuth); beyond 45 degrees the
//!   position draws in to the middle as it rises, reaching the centre at the
//!   zenith; below the floor the same, mirrored;
//! - **distance**: the position is scaled towards the centre in proportion to
//!   the distance (64 sixty-fourths is the reference distance, the cube's
//!   surface; a larger one stays on it).
//!
//! The payload is bare. The encoder wraps it in the Evolution frame and signs
//! it when it has a key; nothing else is signed or written. What is shared
//! with the file encoder (`hz-cli`) is the payload syntax and its writer,
//! [`hz_meta::oamd`], and the identifier the encoder tags the payload with,
//! [`hz_meta::emdf::OAMD`]. The file encoder describes a programme's objects
//! from a master; this describes a frame's from a decoder, so the only code
//! of its own is the mapping from one to the other.

use dca::{SourceRole, SpatialChannel, XMetadata};
use hz_meta::oamd::{
    Block, Distance, Gain, Object, ObjectAudioMetadata, Position, Ramp, Render, Size,
};

use crate::bed::{CARRIED, Slots};

/// Elements of the presentation: the planes the encoder is given.
pub(crate) const ELEMENTS: usize = 16;
/// The elements that carry the bed, the LFE first.
const BED_ELEMENTS: usize = 8;
/// The elements left for extension waveforms.
pub(crate) const EXTENSION_ELEMENTS: usize = ELEMENTS - BED_ELEMENTS;

/// Where each bed element sits in the master's cube, in element order:
/// the LFE (no position), L, R, C, Ls, Rs, Lrs, Rrs. The speakers are on the
/// floor at the corners and the middle of the walls, as a stream's own bed
/// elements declare them.
const BED_AT: [Option<[f64; 3]>; BED_ELEMENTS] = [
    None,
    Some([-1.0, 1.0, 0.0]),
    Some([1.0, 1.0, 0.0]),
    Some([0.0, 1.0, 0.0]),
    Some([-1.0, 0.0, 0.0]),
    Some([1.0, 0.0, 0.0]),
    Some([-1.0, -1.0, 0.0]),
    Some([1.0, -1.0, 0.0]),
];

/// The speaker azimuths (degrees, negative to the left) and the floor point
/// of the cube each lands on, from the rear centre round to the rear centre.
const ANCHORS: [(f64, [f64; 2]); 9] = [
    (-180.0, [0.0, -1.0]),
    (-150.0, [-1.0, -1.0]),
    (-90.0, [-1.0, 0.0]),
    (-30.0, [-1.0, 1.0]),
    (0.0, [0.0, 1.0]),
    (30.0, [1.0, 1.0]),
    (90.0, [1.0, 0.0]),
    (150.0, [1.0, -1.0]),
    (180.0, [0.0, -1.0]),
];
/// The elevation of the height layer, where z reaches 1.
const HEIGHT_DEGREES: f64 = 45.0;

/// Builds payloads, keeping its scratch between frames.
#[derive(Default)]
pub(crate) struct Author {
    blocks: Vec<Block>,
    objects: Vec<Object>,
}

impl Author {
    pub(crate) fn new() -> Self {
        Self {
            blocks: Vec::with_capacity(1),
            objects: Vec::with_capacity(ELEMENTS),
        }
    }

    /// Forget a half-built payload, as after a panic.
    pub(crate) fn clear(&mut self) {
        self.blocks.clear();
        self.objects.clear();
    }

    /// The payload for a decoded frame, with `waveforms` extension waveforms
    /// decoded beside it, parsed here; see [`Self::describe`] for the rest.
    #[cfg(test)]
    pub(crate) fn payload(
        &mut self,
        frame: &dca::HdFrame,
        waveforms: usize,
        slots: &Slots,
    ) -> Option<Vec<u8>> {
        self.describe(parse(frame, waveforms).as_ref(), slots)
    }

    /// The payload that describes the sixteen elements: the bed always, and
    /// each extension element that carries a waveform (`slots`) at the place
    /// `meta` states for it.
    ///
    /// The bytes are the payload itself. The encoder wraps them in the
    /// Evolution frame (and signs it, if it was given a key), so a caller
    /// hands them to `Encoder::set_evolution_at` under [`hz_meta::emdf::OAMD`]
    /// and does nothing else to them.
    pub(crate) fn describe(&mut self, meta: Option<&XMetadata>, slots: &Slots) -> Option<Vec<u8>> {
        self.objects.clear();
        for at in BED_AT {
            self.objects.push(match at {
                // The low frequency channel is a bed and goes where the bed
                // goes.
                None => Object {
                    active: true,
                    gain: Gain::Unity,
                    render: None,
                },
                Some(at) => Object {
                    active: true,
                    gain: Gain::Unity,
                    render: Some(Render {
                        snap: true,
                        ..render(at)
                    }),
                },
            });
        }
        for slot in slots.iter().take(EXTENSION_ELEMENTS) {
            let source = meta.zip(*slot).and_then(|(meta, feed)| meta.source(feed));
            self.objects.push(match source {
                Some(source) => Object {
                    active: true,
                    gain: Gain::Unity,
                    render: Some(render(position_of(source.role))),
                },
                // A plane with no waveform behind it is silent: say it is
                // not there.
                None => Object {
                    active: false,
                    gain: Gain::Silent,
                    render: None,
                },
            });
        }
        debug_assert_eq!(CARRIED, EXTENSION_ELEMENTS);
        // The scratch goes through the metadata and comes back, so its
        // buffers are reused for the next frame.
        self.blocks.clear();
        self.blocks.push(Block {
            offset: 0,
            ramp: Ramp::None,
            objects: std::mem::take(&mut self.objects),
        });
        let oamd = ObjectAudioMetadata {
            lfe: true,
            blocks: std::mem::take(&mut self.blocks),
        };
        let bytes = oamd.write().ok();
        let mut blocks = oamd.blocks;
        if let Some(block) = blocks.pop() {
            self.objects = block.objects;
        }
        blocks.clear();
        self.blocks = blocks;
        bytes
    }
}

/// The metadata a decoded frame states for its `waveforms` extension
/// waveforms, when it has any and it parses.
pub(crate) fn parse(frame: &dca::HdFrame, waveforms: usize) -> Option<XMetadata> {
    if !(frame.x_present || frame.x_imax) || frame.x_payload.is_empty() || waveforms == 0 {
        return None;
    }
    XMetadata::parse(&frame.x_payload, waveforms).ok()
}

/// Where a source sits, as the master's cartesian cube states it (see the
/// module documentation for the mapping).
fn position_of(role: SourceRole) -> [f64; 3] {
    match role {
        SourceRole::Object { position, .. } => cube_point(
            position.azimuth_degrees(),
            position.elevation_degrees(),
            position.distance(),
        ),
        SourceRole::Height(channel) => match channel {
            SpatialChannel::TopFrontLeft => [-1.0, 1.0, 1.0],
            SpatialChannel::TopFrontRight => [1.0, 1.0, 1.0],
            SpatialChannel::TopBackLeft => [-1.0, -1.0, 1.0],
            SpatialChannel::TopBackRight => [1.0, -1.0, 1.0],
            SpatialChannel::TopFrontCenter => [0.0, 1.0, 1.0],
            SpatialChannel::TopSideLeft => [-1.0, 0.0, 1.0],
            SpatialChannel::TopSideRight => [1.0, 0.0, 1.0],
            SpatialChannel::WideLeft => [-1.0, 0.5, 0.0],
            SpatialChannel::WideRight => [1.0, 0.5, 0.0],
        },
    }
}

/// The floor point of the cube for an azimuth in degrees, between the
/// anchors.
fn floor_point(azimuth: f64) -> [f64; 2] {
    // Into (-180, 180].
    let azimuth = (azimuth + 180.0).rem_euclid(360.0) - 180.0;
    let azimuth = if azimuth == -180.0 { 180.0 } else { azimuth };
    for pair in ANCHORS.windows(2) {
        let ((from, a), (to, b)) = (pair[0], pair[1]);
        if (from..=to).contains(&azimuth) {
            let t = (azimuth - from) / (to - from);
            return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
        }
    }
    [0.0, 1.0]
}

/// The master's cube point of a position in degrees and reference distances.
fn cube_point(azimuth: f64, elevation: f64, distance: f64) -> [f64; 3] {
    let [x, y] = floor_point(azimuth);
    let up = (elevation / HEIGHT_DEGREES).clamp(-1.0, 1.0);
    // Past the height layer the position draws in towards the axis.
    let beyond = ((elevation.abs() - HEIGHT_DEGREES) / (90.0 - HEIGHT_DEGREES)).clamp(0.0, 1.0);
    let draw = (1.0 - beyond) * distance.clamp(0.0, 1.0);
    let z = up * distance.clamp(0.0, 1.0);
    [x * draw, y * draw, z]
}

fn render(at: [f64; 3]) -> Render {
    Render {
        position: Position::from_master(at),
        distance: Distance::Unstated,
        zones: 0,
        elevation: at[2].abs() > f64::EPSILON,
        size: Size::Point,
        screen: None,
        snap: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dca::SourceMetadata;
    use dca::{BedFold, FoldPlan, HdFrame, SphericalPosition};

    /// The slots of a frame whose folds are all stated.
    fn slots_of(meta: &XMetadata) -> Slots {
        crate::bed::slots(
            meta,
            &FoldPlan::from_metadata(meta),
            crate::bed::dropped(meta),
        )
    }

    /// Fields written bit by bit, as the extension's metadata is.
    struct Bits(Vec<u8>);

    impl Bits {
        fn push(&mut self, value: u32, width: usize) -> &mut Self {
            for bit in (0..width).rev() {
                self.0.push(((value >> bit) & 1) as u8);
            }
            self
        }

        fn bytes(&mut self) -> Vec<u8> {
            self.0.resize(self.0.len().div_ceil(8) * 8, 0);
            self.0
                .chunks(8)
                .map(|byte| byte.iter().fold(0u8, |acc, bit| acc << 1 | bit))
                .collect()
        }
    }

    fn crc16(data: &[u8]) -> u16 {
        let mut crc = 0xffffu16;
        for &byte in data {
            crc ^= (byte as u16) << 8;
            for _ in 0..8 {
                crc = if crc & 0x8000 != 0 {
                    (crc << 1) ^ 0x1021
                } else {
                    crc << 1
                };
            }
        }
        crc
    }

    /// The extension payload of an alternate profile with one object and the
    /// four fixed heights (five waveforms): the object-record element, the
    /// height-row element, their checksum and the outer control that
    /// follows. The object sits at `azimuth`, `elevation` and `distance`
    /// codes; each height is folded into one bed column.
    fn alternate_payload(azimuth: u32, elevation: u32, distance: u32) -> Vec<u8> {
        const UNITY: u32 = 61;
        let mut w = Bits(Vec::new());
        // The object-record element: header, reference layout (7.1), one
        // declaration, then its position record.
        w.push(0xf1, 8)
            .push(0x40, 8)
            .push(0, 4)
            .push(0, 1)
            .push(1, 4)
            .push(1, 1)
            .push(0, 1)
            .push(1, 1)
            .push(0, 4);
        w.push(0, 1)
            .push(0, 1)
            .push(1, 1)
            .push(1, 1)
            .push(0, 2)
            .push(0, 3);
        w.push(2, 3).push(0x084b, 12).push(0, 8);
        w.push(1, 1).push(3, 2).push(0, 3).push(1, 2).push(0, 10);
        w.push(0, 3)
            .push(0x20, 7)
            .push(1, 1)
            .push(1, 1)
            .push(0, 2)
            .push(UNITY, 6);
        w.push(distance, 6)
            .push(azimuth, 8)
            .push(elevation, 7)
            .push(0, 1);
        w.push(0, 1).push(0, 1); // no reference row
        w.push(0, 1); // no auxiliary layouts
        let mut prefix = w.bytes();

        // The height-row element over the 7.1 plus the four heights: one
        // row per height, naming the height's own column and one bed column.
        let mut w = Bits(Vec::new());
        w.push(3, 8)
            .push(0, 8)
            .push(0, 1)
            .push(1, 4)
            .push(1, 4)
            .push(0, 1)
            .push(0, 1)
            .push(0, 4);
        w.push(1, 1)
            .push(UNITY, 6)
            .push(0, 1)
            .push(3, 3)
            .push(0x084b | 0x8020, 16);
        w.push(2, 5)
            .push(0, 6)
            .push(1, 1)
            .push(0, 1)
            .push(0x3fa, 12);
        // Columns in order: C, L, R, LFE, TFL, TFR, Lsr, Rsr, Lss, Rss, TBL, TBR.
        for (own, bed) in [(4, 1), (5, 2), (10, 6), (11, 7)] {
            let mask = 1u32 << own | 1 << bed;
            w.push(mask, 12);
            for column in 0..12 {
                if mask & (1 << column) != 0 {
                    w.push(if column == own { UNITY } else { 55 }, 6);
                }
            }
        }
        prefix.extend_from_slice(&w.bytes());

        let crc = crc16(&prefix);
        prefix.extend_from_slice(&crc.to_be_bytes());
        // The outer control: channel sets present, a marker, then padding.
        prefix.extend_from_slice(&[0x03, 0x34, 0x38, 0x8c, 0x4f, 0x00]);
        prefix.extend_from_slice(&[0xb2, 0, 0, 0, 0, 0, 0, 0]);
        prefix
    }

    fn frame_with(x_present: bool, x_imax: bool, x_payload: Vec<u8>) -> HdFrame {
        HdFrame {
            x_present,
            x_imax,
            x_payload,
            ..HdFrame::default()
        }
    }

    fn read(bytes: &[u8]) -> ObjectAudioMetadata {
        ObjectAudioMetadata::read(bytes).expect("reads back")
    }

    #[test]
    fn metadata_that_does_not_parse_describes_the_bed_alone() {
        let mut author = Author::new();
        for frame in [
            frame_with(true, false, vec![]),
            frame_with(false, true, vec![0xff; 7]),
        ] {
            let bytes = author.payload(&frame, 4, &[None; 8]).expect("bed");
            assert_bed_only(&bytes);
        }
    }

    /// The alternate profiles are the ones the decoder flags `x_imax` and
    /// they carry an object record with a position: they get the full
    /// description, element for element.
    #[test]
    fn an_alternate_profile_frame_is_described_element_by_element() {
        // Azimuth code 100 is 30 degrees to the left, elevation code 100 is
        // 60 degrees up, distance 63 is the reference distance.
        let payload = alternate_payload(100, 100, 63);
        assert!(
            XMetadata::parse(&payload, 5).is_ok(),
            "the synthetic payload parses as the alternate profile"
        );
        let mut author = Author::new();
        let frame = frame_with(false, true, payload);
        let meta = parse(&frame, 5).expect("parses");
        let bytes = author
            .payload(&frame, 5, &slots_of(&meta))
            .expect("an x_imax frame with an object record gets metadata");
        let metadata = read(&bytes);
        assert!(metadata.lfe);
        assert_eq!(metadata.objects(), ELEMENTS);
        let elements = &metadata.blocks[0].objects;

        // Element 0 is the LFE, 1 to 7 the bed, parked at their speakers.
        assert!(elements[0].active && elements[0].render.is_none());
        let bed = [
            [-1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
        ];
        for (element, at) in bed.iter().enumerate() {
            let render = elements[element + 1]
                .render
                .expect("a bed element is placed");
            assert_eq!(render.position, Position::from_master(*at), "{element}");
            assert!(render.snap && !render.elevation);
        }

        // Element 8 is the object, where the record put it: 30 degrees to
        // the left is the left speaker's azimuth, and 60 degrees up is past
        // the height layer (45), so it is drawn a third of the way in.
        let at = Position::from_master([-2.0 / 3.0, 2.0 / 3.0, 1.0]);
        assert_eq!(elements[8].render.expect("placed").position, at);
        // 9 to 12 are the heights, 13 to 15 have no waveform.
        let heights = [
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
        ];
        for (i, at) in heights.iter().enumerate() {
            let render = elements[9 + i].render.expect("a height is placed");
            assert_eq!(render.position, Position::from_master(*at), "height {i}");
            assert!(render.elevation);
        }
        for element in &elements[13..] {
            assert!(!element.active && element.render.is_none());
        }
    }

    /// The elements the bed alone describes: the LFE and the seven bed
    /// channels there, nothing on the extension elements.
    fn assert_bed_only(bytes: &[u8]) {
        let metadata = read(bytes);
        assert!(metadata.lfe);
        let elements = &metadata.blocks[0].objects;
        assert_eq!(elements.len(), ELEMENTS);
        assert!(elements[..8].iter().all(|e| e.active));
        assert!(
            elements[8..]
                .iter()
                .all(|e| !e.active && e.render.is_none())
        );
    }

    /// Neither flag, no extension: the bed is described all the same, so a
    /// decoder never has to hold the last description.
    #[test]
    fn a_frame_with_neither_flag_describes_the_bed_alone() {
        let mut author = Author::new();
        let payload = alternate_payload(100, 100, 63);
        for (frame, waveforms) in [
            (frame_with(false, false, payload), 5),
            (frame_with(false, false, Vec::new()), 0),
        ] {
            assert!(parse(&frame, waveforms).is_none());
            let bytes = author.payload(&frame, waveforms, &[None; 8]).expect("bed");
            assert_bed_only(&bytes);
        }
    }

    fn height(channel: SpatialChannel) -> SourceMetadata {
        SourceMetadata {
            role: SourceRole::Height(channel),
            fold: BedFold::Known([0.0; 8]),
        }
    }

    fn object(azimuth_half_degrees: i16) -> SourceMetadata {
        SourceMetadata {
            role: SourceRole::Object {
                position: SphericalPosition {
                    azimuth_half_degrees,
                    elevation_half_degrees: 0,
                    distance_64ths: 64,
                },
                centre_height_alternative: false,
            },
            fold: BedFold::Unknown,
        }
    }

    const FOUR_HEIGHTS: [SpatialChannel; 4] = [
        SpatialChannel::TopFrontLeft,
        SpatialChannel::TopFrontRight,
        SpatialChannel::TopBackLeft,
        SpatialChannel::TopBackRight,
    ];

    #[test]
    fn the_standard_four_heights_leave_the_last_four_elements_unused() {
        let sources: Vec<_> = FOUR_HEIGHTS.iter().map(|c| height(*c)).collect();
        let meta = XMetadata::from_sources(&sources).expect("four sources");
        let bytes = Author::new()
            .describe(Some(&meta), &slots_of(&meta))
            .expect("described");
        let elements = &read(&bytes).blocks[0].objects;
        assert_eq!(elements.len(), ELEMENTS);
        assert!(
            elements[8..12]
                .iter()
                .all(|e| e.active && e.render.is_some())
        );
        assert!(elements[12..].iter().all(|e| !e.active));
    }

    /// The widest profile has nine waveforms and the stream room for eight:
    /// with no speaker object to leave in the bed the last height is the one
    /// left out, as the planes do.
    #[test]
    fn a_ninth_waveform_is_not_described() {
        let mut sources: Vec<_> = (0..5).map(|i| object(-60 + 30 * i)).collect();
        sources.extend(FOUR_HEIGHTS.iter().map(|c| height(*c)));
        let meta = XMetadata::from_sources(&sources).expect("nine sources");
        // The objects here state no fold, so the cleaner would not carry
        // them; this is about the description, so say all eight are.
        let slots: Slots = std::array::from_fn(Some);
        let bytes = Author::new()
            .describe(Some(&meta), &slots)
            .expect("described");
        let elements = &read(&bytes).blocks[0].objects;
        assert_eq!(elements.len(), ELEMENTS);
        assert!(elements[8..].iter().all(|e| e.active));
        // The eighth waveform is the third height; the fourth is left out.
        let third = Position::from_master([-1.0, -1.0, 1.0]);
        assert_eq!(elements[15].render.expect("placed").position, third);
    }

    /// A frame with nothing carried describes the bed alone.
    #[test]
    fn a_frame_that_carries_no_waveform_describes_the_bed_alone() {
        let meta = XMetadata::from_sources(&[]).expect("empty");
        let bytes = Author::new()
            .describe(Some(&meta), &[None; 8])
            .expect("described");
        assert_bed_only(&bytes);
    }

    fn at(azimuth: f64, elevation: f64, distance: f64) -> [f64; 3] {
        cube_point(azimuth, elevation, distance)
    }

    /// An object at a speaker's own azimuth lands exactly on the point of
    /// that speaker's element: the same place the stream's fold puts it, at
    /// gain 1.
    #[test]
    fn an_object_at_a_speaker_lands_on_the_speaker() {
        let speakers = [
            (-30.0, 1),
            (30.0, 2),
            (0.0, 3),
            (-90.0, 4),
            (90.0, 5),
            (-150.0, 6),
            (150.0, 7),
        ];
        for (azimuth, element) in speakers {
            assert_eq!(
                Some(at(azimuth, 0.0, 1.0)),
                BED_AT[element],
                "{azimuth} degrees"
            );
        }
        // Behind, the middle of the rear wall; either way round.
        assert_eq!(at(180.0, 0.0, 1.0), [0.0, -1.0, 0.0]);
        assert_eq!(at(-180.0, 0.0, 1.0), [0.0, -1.0, 0.0]);
        // And through the stream's own units (half degrees, 64ths).
        let object = SourceRole::Object {
            position: SphericalPosition {
                azimuth_half_degrees: -60,
                elevation_half_degrees: 0,
                distance_64ths: 64,
            },
            centre_height_alternative: false,
        };
        assert_eq!(position_of(object), [-1.0, 1.0, 0.0]);
    }

    /// Half way between two speakers is half way between their points.
    #[test]
    fn the_positions_between_speakers_are_on_the_walls_between() {
        assert_eq!(at(-60.0, 0.0, 1.0), [-1.0, 0.5, 0.0]);
        assert_eq!(at(-15.0, 0.0, 1.0), [-0.5, 1.0, 0.0]);
        assert_eq!(at(15.0, 0.0, 1.0), [0.5, 1.0, 0.0]);
        assert_eq!(at(60.0, 0.0, 1.0), [1.0, 0.5, 0.0]);
        assert_eq!(at(120.0, 0.0, 1.0), [1.0, -0.5, 0.0]);
        assert_eq!(at(-120.0, 0.0, 1.0), [-1.0, -0.5, 0.0]);
        assert_eq!(at(165.0, 0.0, 1.0), [0.5, -1.0, 0.0]);
    }

    /// The height layer is 45 degrees up: above a speaker's azimuth it is the
    /// corner at the top, the same point the fixed heights are placed at.
    #[test]
    fn the_height_elevation_lands_at_the_height_positions() {
        let corners = [
            (-30.0, SpatialChannel::TopFrontLeft),
            (30.0, SpatialChannel::TopFrontRight),
            (-150.0, SpatialChannel::TopBackLeft),
            (150.0, SpatialChannel::TopBackRight),
            (0.0, SpatialChannel::TopFrontCenter),
            (-90.0, SpatialChannel::TopSideLeft),
            (90.0, SpatialChannel::TopSideRight),
        ];
        for (azimuth, channel) in corners {
            assert_eq!(
                at(azimuth, 45.0, 1.0),
                position_of(SourceRole::Height(channel)),
                "{channel:?}"
            );
        }
        // Half way up is half way to the ceiling; the zenith is the middle
        // of it; below the floor mirrors.
        assert_eq!(at(-30.0, 22.5, 1.0), [-1.0, 1.0, 0.5]);
        assert_eq!(at(-30.0, 90.0, 1.0), [0.0, 0.0, 1.0]);
        assert_eq!(at(-30.0, -45.0, 1.0), [-1.0, 1.0, -1.0]);
    }

    /// Nearer than the reference distance is nearer the middle; farther stays
    /// on the cube.
    #[test]
    fn distance_scales_the_position_towards_the_middle() {
        assert_eq!(at(-30.0, 0.0, 0.5), [-0.5, 0.5, 0.0]);
        assert_eq!(at(-30.0, 0.0, 2.0), [-1.0, 1.0, 0.0]);
        assert_eq!(at(-30.0, 0.0, 0.0), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn every_height_has_a_place_in_the_cube() {
        for channel in [
            SpatialChannel::TopFrontLeft,
            SpatialChannel::TopFrontRight,
            SpatialChannel::TopBackLeft,
            SpatialChannel::TopBackRight,
            SpatialChannel::TopFrontCenter,
            SpatialChannel::TopSideLeft,
            SpatialChannel::TopSideRight,
            SpatialChannel::WideLeft,
            SpatialChannel::WideRight,
        ] {
            let at = position_of(SourceRole::Height(channel));
            assert!(
                at.iter().all(|v| (-1.0..=1.0).contains(v)),
                "{channel:?}: {at:?}"
            );
        }
    }
}
