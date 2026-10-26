//! `decode(encode(x)) == x`, for every shape the encoder writes.
//!
//! The one question worth asking of a lossless codec, asked without an
//! external decoder installed: every presentation of every stream, through the
//! reference decoder in [`decoder`], compared against what went in.
//!
//! Fixtures are generated here rather than read from disk — a seeded generator
//! and a few closed forms — so the suite runs anywhere and a failure can be
//! reproduced from its seed alone. What real material proves that these cannot
//! is still `cargo xtask mlp`, which is the oracle; this is what runs on every
//! `cargo test`.

mod decoder;

use hz_mlp::format::{DynamicRange, RestartSync};
use hz_mlp::{Config, Effort, Encoder, SampleBits, reader};

/// Encode channel-major input and check every presentation decodes back to it.
///
/// The input is in a WAV file's channel order, which is not the format's past
/// 5.1 — so the comparison goes through the encoder's own permutation, which
/// is what a decoder reports.
#[track_caller]
fn round_trip(config: Config, planes: &[Vec<i32>], effort: Effort) -> Vec<u8> {
    let mut encoder = Encoder::new(config).expect("a shape the encoder writes");
    encoder.set_effort(effort);
    let stream = encode(&mut encoder, planes);
    check(&encoder, &stream, planes, config);
    stream
}

/// Feed the planes in one access unit at a time, with a final short one where
/// the length is not a whole number of units.
fn encode(encoder: &mut Encoder, planes: &[Vec<i32>]) -> Vec<u8> {
    let channels = planes.len();
    let frames = planes[0].len();
    assert!(
        planes.iter().all(|plane| plane.len() == frames),
        "every channel has the same length"
    );
    let unit = encoder.frame_size();
    let mut stream = Vec::new();
    let mut at = 0;
    while at + unit <= frames {
        stream.extend_from_slice(&encoder.push(&interleave(planes, at, unit)));
        at += unit;
    }
    stream.extend_from_slice(&encoder.finish(&interleave(planes, at, frames - at)));
    let _ = channels;
    stream
}

fn interleave(planes: &[Vec<i32>], at: usize, frames: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(frames * planes.len());
    for frame in at..at + frames {
        for plane in planes {
            out.push(plane[frame]);
        }
    }
    out
}

/// Every presentation the stream declares, decoded and compared.
#[track_caller]
fn check(encoder: &Encoder, stream: &[u8], planes: &[Vec<i32>], config: Config) {
    let shift = match config.bits {
        SampleBits::Sixteen => 8,
        SampleBits::TwentyFour => 0,
    };
    let order = encoder.channel_order();
    let frames = planes[0].len();
    let substreams = decoder::substreams(stream).expect("the stream opens with a major sync");

    for presentation in 0..substreams {
        let decoded = decoder::decode(stream, presentation)
            .unwrap_or_else(|e| panic!("presentation {presentation}: {e}"));
        assert_eq!(
            decoded.frames(),
            frames,
            "presentation {presentation} came back {} frames long against {frames}",
            decoded.frames()
        );
        for (frame, got) in decoded.samples.chunks_exact(decoded.channels).enumerate() {
            for (channel, sample) in got.iter().enumerate() {
                let wanted = planes[order[channel]][frame] << shift;
                assert_eq!(
                    *sample, wanted,
                    "presentation {presentation}, frame {frame}, channel {channel}"
                );
            }
        }
    }
}

/// A seeded generator, so a failure is reproducible from its seed alone.
struct Noise(u64);

impl Noise {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A sample of `bits` bits, signed.
    fn sample(&mut self, bits: u32) -> i32 {
        let raw = (self.next() >> 24) as i32;
        (raw << (32 - bits)) >> (32 - bits)
    }
}

fn tone(frames: usize, period: f64, amplitude: f64) -> Vec<i32> {
    (0..frames)
        .map(|n| ((n as f64 * std::f64::consts::TAU / period).sin() * amplitude) as i32)
        .collect()
}

fn noise(seed: u64, frames: usize, bits: u32) -> Vec<i32> {
    let mut source = Noise::new(seed);
    (0..frames).map(|_| source.sample(bits)).collect()
}

fn config(channels: usize, rate: u32, bits: SampleBits) -> Config {
    Config {
        sample_rate: rate,
        channels,
        bits,
    }
}

/// Every channel count the encoder accepts, through every presentation it
/// declares.
///
/// The counts are what the arrangement fields can name — one, two, six, eight
/// — and then an object programme, whose only degree of freedom is where the
/// fourth substream ends.
#[test]
fn every_channel_count_round_trips() {
    for channels in [1usize, 2, 6, 8, 9, 12, 16] {
        let frames = 400;
        let planes: Vec<Vec<i32>> = (0..channels)
            .map(|channel| {
                if channel % 3 == 0 {
                    tone(frames, 17.0 + channel as f64, 3_000_000.0)
                } else {
                    noise(0x9e37_79b9 + channel as u64, frames, 20)
                }
            })
            .collect();
        round_trip(
            config(channels, 48_000, SampleBits::TwentyFour),
            &planes,
            Effort::Full,
        );
    }
}

/// Nine to sixteen elements, every one of them: shipped streams carry twelve,
/// fourteen and sixteen, and the sizes between are written the same way.
#[test]
fn every_element_count_round_trips() {
    for channels in 9..=16usize {
        let frames = 120;
        let planes: Vec<Vec<i32>> = (0..channels)
            .map(|channel| noise(0x2545_f491 + channel as u64, frames, 18))
            .collect();
        round_trip(
            config(channels, 48_000, SampleBits::TwentyFour),
            &planes,
            Effort::Full,
        );
    }
}

/// The six sample rates the format carries. The frame size doubles with each
/// rate factor, so this is also every block length.
#[test]
fn every_sample_rate_round_trips() {
    for rate in [44_100u32, 48_000, 88_200, 96_000, 176_400, 192_000] {
        let planes = vec![tone(700, 23.0, 5_000_000.0), noise(0xdead_beef, 700, 22)];
        round_trip(
            config(2, rate, SampleBits::TwentyFour),
            &planes,
            Effort::Full,
        );
    }
}

/// A matrix that would leave the codec's domain is refused.
///
/// A coding matrix's weight is a least-squares fit over the interval, which
/// takes energy off on average and not on every sample. Two channels near
/// full scale, alike on every unit but one where the second turns over: the
/// fit says "take nearly all of it off", the trials the matrix is costed on
/// never see the odd unit, and on that unit the matrixed channel would be
/// nearly twice full scale — which a decoder reconstructs in twenty-four
/// bits and refuses. The stream has to decode, so the matrix has to go: the
/// test decoder holds the reference's bound, and the encoder says it
/// refused.
#[test]
fn a_matrix_that_would_leave_the_domain_is_refused() {
    // One restart interval of forty-frame units.
    let frames = 5_120;
    let left = tone(frames, 29.0, 8_300_000.0);
    let mut right = left.clone();
    // Unit fifty, which none of the four trial units is.
    for sample in &mut right[2_000..2_040] {
        *sample = -*sample;
    }
    let planes = [left, right];
    let config = config(2, 48_000, SampleBits::TwentyFour);
    let mut encoder = Encoder::new(config).expect("a shape the encoder writes");
    encoder.set_effort(Effort::Full);
    let stream = encode(&mut encoder, &planes);
    check(&encoder, &stream, &planes, config);
    assert!(
        encoder.stats().matrices_refused > 0,
        "the matrix that would have left the domain was refused"
    );
}

/// Sixteen-bit input is shifted into the codec's twenty-four on the way in,
/// and a decoder hands back the shifted value — so what comes out is what went
/// in, eight bits higher.
#[test]
fn sixteen_bit_input_round_trips() {
    let planes = vec![tone(400, 31.0, 20_000.0), noise(0x1234_5678, 400, 16)];
    round_trip(
        config(2, 48_000, SampleBits::Sixteen),
        &planes,
        Effort::Full,
    );
}

/// Content narrower than its container: the encoder takes the dead low bits
/// off and the decoder puts them back, after the filters and after the
/// matrices.
#[test]
fn dead_low_bits_come_back() {
    let planes: Vec<Vec<i32>> = (0..2)
        .map(|channel| {
            noise(0xabc_def + channel, 400, 20)
                .iter()
                .map(|sample| sample << 4)
                .collect()
        })
        .collect();
    let stream = round_trip(
        config(2, 48_000, SampleBits::TwentyFour),
        &planes,
        Effort::Full,
    );
    assert!(!stream.is_empty());
}

/// A silent channel beside a loud one, which is the shape a bed has and the
/// one that makes an encoder want a shift it then has to hand back.
#[test]
fn a_silent_channel_round_trips() {
    let planes = vec![
        vec![0i32; 400],
        tone(400, 13.0, 6_000_000.0),
        vec![0i32; 400],
        noise(0x5bd1_e995, 400, 21),
        vec![0i32; 400],
        vec![0i32; 400],
    ];
    round_trip(
        config(6, 48_000, SampleBits::TwentyFour),
        &planes,
        Effort::Full,
    );
}

/// A correlated pair is what the matrices are for, and a matrix is the one
/// thing that has to be undone by every presentation that reaches the channel
/// it writes.
#[test]
fn a_correlated_pair_round_trips() {
    let left = noise(0x1111_2222, 800, 22);
    let mut source = Noise::new(0x3333_4444);
    let right: Vec<i32> = left
        .iter()
        .map(|sample| sample / 2 + source.sample(16))
        .collect();
    round_trip(
        config(2, 48_000, SampleBits::TwentyFour),
        &[left, right],
        Effort::Full,
    );
}

/// A stream whose length is not a whole number of access units: the last one
/// is padded and says how much of it to throw away.
#[test]
fn a_ragged_length_round_trips() {
    for tail in [1usize, 7, 39] {
        let frames = 400 + tail;
        let planes = vec![
            tone(frames, 19.0, 4_000_000.0),
            noise(0x7777_8888, frames, 20),
        ];
        round_trip(
            config(2, 48_000, SampleBits::TwentyFour),
            &planes,
            Effort::Full,
        );
    }
}

/// Long enough to cross several restart intervals, which is where the filter
/// history is thrown away and the second filter is decided.
#[test]
fn several_restart_intervals_round_trip() {
    for effort in [Effort::Full, Effort::Fast] {
        // Two hundred and sixty units at forty samples: two restarts and a bit,
        // and past the history `pick_second` needs before it will choose one.
        let frames = 260 * 40;
        let planes = vec![
            // A resonance, which is what a second filter is for.
            tone(frames, 4.7, 7_000_000.0),
            noise(0x9999_aaaa, frames, 21),
        ];
        round_trip(config(2, 48_000, SampleBits::TwentyFour), &planes, effort);
    }
}

/// A dynamic range word on every substream, which makes each directory entry
/// four bytes rather than two.
#[test]
fn dynamic_range_words_round_trip() {
    let channels = 12;
    let frames = 200;
    let planes: Vec<Vec<i32>> = (0..channels)
        .map(|channel| noise(0xbbbb_cccc + channel as u64, frames, 19))
        .collect();
    let mut encoder = Encoder::new(config(channels, 48_000, SampleBits::TwentyFour))
        .expect("an object programme");
    for substream in 0..4 {
        encoder.set_dynamic_range(substream, DynamicRange::from_db(-3.6 - substream as f64, 3));
    }
    let stream = encode(&mut encoder, &planes);
    check(
        &encoder,
        &stream,
        &planes,
        config(channels, 48_000, SampleBits::TwentyFour),
    );
}

/// An Evolution payload after the last substream, which is where object
/// metadata rides — and the sample of the unit it takes effect at, which a
/// reference stream states on every payload and a decoder dates it from.
#[test]
fn an_evolution_payload_round_trips() {
    let channels = 16;
    let frames = 200;
    let planes: Vec<Vec<i32>> = (0..channels)
        .map(|channel| noise(0xdddd_eeee + channel as u64, frames, 18))
        .collect();
    let mut encoder = Encoder::new(config(channels, 48_000, SampleBits::TwentyFour))
        .expect("an object programme");

    let unit = encoder.frame_size();
    let mut stream = Vec::new();
    let mut at = 0;
    let mut payload = vec![0u8; 40];
    while at + unit <= frames {
        for (index, byte) in payload.iter_mut().enumerate() {
            *byte = (at + index) as u8;
        }
        // Nought, then five, thirteen, twenty-one and twenty-nine: the four
        // moments a reference stream puts its payloads at, and the start.
        encoder.set_evolution_at(
            11,
            &payload,
            ((at / unit) as u32 * 8 + 5) % 40 * u32::from(at > 0),
        );
        stream.extend_from_slice(&encoder.push(&interleave(&planes, at, unit)));
        at += unit;
    }
    stream.extend_from_slice(&encoder.finish(&interleave(&planes, at, frames - at)));
    check(
        &encoder,
        &stream,
        &planes,
        config(channels, 48_000, SampleBits::TwentyFour),
    );

    // And the block reads back as the reference's does: the payload whole,
    // closed by an eight-bit primary protection field and no secondary one.
    let mut at = 0;
    let mut substreams = 0;
    let mut evolution = false;
    let mut carried = 0;
    while at < stream.len() {
        let unit = reader::access_unit(&stream[at..], substreams, evolution)
            .expect("every access unit reads");
        substreams = unit.directory.len();
        if let Some(sync) = &unit.major_sync {
            evolution = sync.flags & (1 << 12) != 0;
        }
        if let Some(extra) = &unit.evolution {
            assert!(extra.frame_ok && extra.parity_ok && extra.check_nibble_ok);
            assert_eq!(extra.protection, (8, 0));
            assert_eq!(extra.payloads.len(), 1);
            assert_eq!(extra.payloads[0].id, 11);
            assert_eq!(extra.payloads[0].bytes.len(), payload.len());
            // Every unit carried one, so this is the unit's own index.
            let offset = (carried as u32 * 8 + 5) % 40 * u32::from(carried > 0);
            assert_eq!(
                extra.payloads[0].sample_offset,
                (offset > 0).then_some(offset),
                "unit {carried} takes effect where it was told to"
            );
            carried += 1;
        }
        at += unit.bytes;
    }
    assert!(carried > 0, "the payload rode in at least one unit");
}

/// No early substream declares more matrices than every decoder reads.
///
/// The count is a four-bit field and restart sync words A and B can say
/// fifteen, but the decoder the world embeds — FFmpeg's, in every player that
/// is not Dolby's — refuses a TrueHD substream past eight, and the reference
/// streams write two, six, eight and nine. Written past it a stream is refused
/// unit by unit, and the errors cascade into filter orders, quantiser steps
/// and sync words that were never wrong. A 7.1's presentation rows already
/// take the eight, so the coding matrices that alike elements invite have to
/// give way — which is what this checks, on elements alike enough that every
/// one of them would take one.
#[test]
fn no_early_substream_declares_more_matrices_than_every_decoder_reads() {
    use hz_mlp::hierarchy::Presentation;
    const ELEMENTS: usize = 12;
    // Two restart intervals of forty-frame units.
    let frames = 10_240;
    // Every element nearly the same signal, at its own level, which is what
    // makes a coding matrix pay on every channel — noise rather than a tone,
    // since a tone the filters predict on their own leaves nothing for a
    // matrix to take off.
    let base = noise(0xba5e_ba5e, frames, 20);
    let planes: Vec<Vec<i32>> = (0..ELEMENTS)
        .map(|element| {
            let own = noise(0x51ed_c0de + element as u64, frames, 12);
            base.iter()
                .zip(&own)
                .map(|(shared, own)| shared * (element as i32 + 4) / 8 + own)
                .collect()
        })
        .collect();
    let row = |gains: &[(usize, f64)]| {
        let mut out = vec![0.0; ELEMENTS];
        for (element, gain) in gains {
            out[*element] = *gain;
        }
        out
    };
    let mix = |rows: &[(&[f64], f64)]| {
        let mut out = vec![0.0; ELEMENTS];
        for (row, gain) in rows {
            for (slot, value) in out.iter_mut().zip(*row) {
                *slot += gain * value;
            }
        }
        out
    };
    // A 7.1 of eight rows — the eight a substream may carry — each a bed
    // element with an object folded in, and the narrower ones folded from it.
    let seven: Vec<Vec<f64>> = (0..8)
        .map(|channel| row(&[(channel, 1.0), (8 + channel % 4, 0.5)]))
        .collect();
    let five = vec![
        seven[0].clone(),
        seven[1].clone(),
        seven[2].clone(),
        seven[3].clone(),
        mix(&[(&seven[4], 1.0), (&seven[6], 0.87)]),
        mix(&[(&seven[5], 1.0), (&seven[7], 0.87)]),
    ];
    let two = vec![
        mix(&[(&five[0], 0.5), (&five[2], 0.354), (&five[4], 0.5)]),
        mix(&[(&five[1], 0.5), (&five[2], 0.354), (&five[5], 0.5)]),
    ];
    let elements: Vec<Vec<f64>> = (0..ELEMENTS)
        .map(|element| row(&[(element, 1.0)]))
        .collect();
    let presentations: Vec<Presentation> = [(2usize, two), (6, five), (8, seven), (12, elements)]
        .into_iter()
        .map(|(channels, rows)| Presentation { channels, rows })
        .collect();

    let config = config(ELEMENTS, 48_000, SampleBits::TwentyFour);
    let mut encoder = Encoder::new(config).expect("twelve elements");
    encoder.set_presentations(&presentations);
    let stream = encode(&mut encoder, &planes);

    // The elements come back exactly, and every narrower presentation reads.
    let full = decoder::decode(&stream, 3).expect("the elements decode");
    assert_eq!(full.channels, ELEMENTS);
    for (frame, got) in full.samples.chunks_exact(ELEMENTS).enumerate() {
        for (element, sample) in got.iter().enumerate() {
            assert_eq!(
                *sample, planes[element][frame],
                "frame {frame}, element {element}"
            );
        }
    }
    for which in 0..3 {
        decoder::decode(&stream, which).unwrap_or_else(|e| panic!("presentation {which}: {e}"));
    }

    let mut at = 0;
    let mut substreams = 0;
    let mut evolution = false;
    let mut widest_early = 0;
    while at < stream.len() {
        let unit = reader::access_unit(&stream[at..], substreams, evolution)
            .expect("every access unit reads");
        substreams = unit.directory.len();
        if let Some(sync) = &unit.major_sync {
            evolution = sync.flags & (1 << 12) != 0;
        }
        for (which, substream) in unit.substreams.iter().enumerate() {
            let Some(restart) = &substream.restart else {
                continue;
            };
            let most = if matches!(restart.sync, RestartSync::C) {
                16
            } else {
                widest_early = widest_early.max(substream.matrices.len());
                8
            };
            assert!(
                substream.matrices.len() <= most,
                "substream {which} declares {} matrices under sync word {:#06x}",
                substream.matrices.len(),
                restart.sync.word()
            );
        }
        at += unit.bytes;
    }
    // Not vacuously: the 7.1's rows alone fill a substream to the bound.
    assert_eq!(widest_early, 8, "an early substream reached the bound");
}

/// Seven point one, where the format's channel order and a WAV file's differ:
/// the sides go on the wire before the rears. An encoder that ignored the
/// difference would produce a stream whose surrounds are swapped and whose
/// every check passes.
#[test]
fn the_seven_one_permutation_round_trips() {
    let frames = 400;
    // Each channel a different constant, so a permutation cannot hide.
    let planes: Vec<Vec<i32>> = (1..=8)
        .map(|channel| vec![channel * 100_000; frames])
        .collect();
    let mut encoder =
        Encoder::new(config(8, 48_000, SampleBits::TwentyFour)).expect("eight channels");
    let stream = encode(&mut encoder, &planes);
    assert_eq!(
        encoder.channel_order(),
        &[0, 1, 2, 3, 6, 7, 4, 5],
        "the sides go on the wire before the rears"
    );
    check(
        &encoder,
        &stream,
        &planes,
        config(8, 48_000, SampleBits::TwentyFour),
    );
}

/// Two hundred short streams from two hundred seeds, which is what finds the
/// shapes nobody thought to write a fixture for.
#[test]
fn randomised_short_streams_round_trip() {
    for seed in 0..200u64 {
        let mut source = Noise::new(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x5555);
        let channels = [1usize, 2, 6, 8][(source.next() % 4) as usize];
        let frames = 40 + (source.next() % 130) as usize;
        let width = 8 + (source.next() % 17) as u32;
        let planes: Vec<Vec<i32>> = (0..channels)
            .map(|channel| {
                let mut plane = Vec::with_capacity(frames);
                let mut own = Noise::new(seed ^ (channel as u64) << 32 ^ 0xa5a5);
                for _ in 0..frames {
                    plane.push(own.sample(width));
                }
                plane
            })
            .collect();
        let planes = if seed % 5 == 0 {
            // One stream in five is silent on its last channel, which is the
            // case that asks for a shift and then has to hand it back.
            let mut planes = planes;
            let last = planes.len() - 1;
            planes[last].fill(0);
            planes
        } else {
            planes
        };
        round_trip(
            config(channels, 48_000, SampleBits::TwentyFour),
            &planes,
            if seed % 2 == 0 {
                Effort::Full
            } else {
                Effort::Fast
            },
        );
    }
}

/// The suite must not be able to pass vacuously.
///
/// One byte of one residual flipped, and the lossless check the next restart
/// header carries no longer matches what the samples fold to — which is the
/// format's own way of saying a stream did not decode to what was encoded.
#[test]
fn a_corrupt_residual_is_caught() {
    let frames = 6 * 40;
    let planes = vec![
        tone(frames, 23.0, 5_000_000.0),
        noise(0x0f0f_0f0f, frames, 22),
    ];
    let mut encoder =
        Encoder::new(config(2, 48_000, SampleBits::TwentyFour)).expect("two channels");
    let stream = encode(&mut encoder, &planes);
    decoder::decode(&stream, 0).expect("the stream as written decodes");

    // Well past the first unit's header and into block data. Every byte tried,
    // because some of them land in padding or in a field whose corruption the
    // structural checks catch first — what matters is that a corruption which
    // parses cleanly is still caught.
    let mut caught = 0;
    let mut parsed_but_wrong = 0;
    for at in 40..stream.len().min(400) {
        let mut damaged = stream.clone();
        damaged[at] ^= 0x01;
        match decoder::decode(&damaged, 0) {
            Err(_) => caught += 1,
            Ok(decoded) => {
                // It decoded to *something*; it must not be the same something.
                let same = decoded.frames() == frames
                    && (0..frames).all(|frame| {
                        decoded.frame(frame)[0] == planes[0][frame]
                            && decoded.frame(frame)[1] == planes[1][frame]
                    });
                assert!(!same, "flipping a bit at {at} changed nothing");
                parsed_but_wrong += 1;
            }
        }
    }
    assert!(caught > 0, "no corruption was refused outright");
    let _ = parsed_but_wrong;
}

/// A correlation that drifts, which is what makes a matrix carry a step: the
/// decoder ramps the coefficient across the access unit and adds the whole
/// step at the end of it. Only the immersive syntax can say this, and only for
/// a channel no earlier substream describes.
///
/// Long enough for the aiming to have a previous interval to extrapolate from,
/// which is what a stream has to cross two restarts to get.
#[test]
fn a_matrix_that_moves_round_trips() {
    const CHANNELS: usize = 16;
    let frames = 400 * 40;
    let mut source = Noise::new(0x853c_49e6_748f_ea9b);
    let mut planes: Vec<Vec<i32>> = (0..CHANNELS).map(|_| Vec::with_capacity(frames)).collect();
    for frame in 0..frames {
        for pair in 0..CHANNELS / 2 {
            let value = source.sample(20);
            // A source and that source scaled by a weight that sweeps, which
            // is a correlation a step can track.
            let weight = (frame as f64 / 9_000.0 + pair as f64).sin() * 0.9;
            planes[pair * 2].push(value);
            planes[pair * 2 + 1].push((f64::from(value) * weight) as i32);
        }
    }
    round_trip(
        config(CHANNELS, 48_000, SampleBits::TwentyFour),
        &planes,
        Effort::Full,
    );
}

/// A hierarchy of presentations: the narrow ones are folds of the elements,
/// and the widest is the elements themselves, bit for bit.
///
/// The presentations are consistent folds — the 2.0 folded out of the 5.1
/// and the 5.1 out of the 7.1, as a real programme's are — since each has to
/// lie in the span of the channels its substream reads. Two elements are
/// silent and one is loud, which is what puts the loud one under a stereo row
/// and the silent ones in the last substream; and the elements come back in
/// order through the last substream's channel assignment, which is a
/// permutation that is not its own inverse.
#[test]
fn a_hierarchy_of_presentations_round_trips() {
    let (planes, presentations) = fold_programme();
    let config = config(planes.len(), 48_000, SampleBits::TwentyFour);
    let mut encoder = Encoder::new(config).expect("twelve elements");
    encoder.set_presentations(&presentations);
    let stream = encode(&mut encoder, &planes);
    let substreams = decoder::substreams(&stream).expect("a major sync");
    assert_eq!(substreams, 4);
    let stats = encoder.stats();
    assert_eq!(stats.folds_asked, 2, "two restart intervals, both asked");
    assert_eq!(
        stats.folds_carried, stats.folds_asked,
        "every interval carries its folds"
    );
    assert_folds_hold(&stream, &planes, &presentations);
}

/// Decorrelating the hierarchy's early channels takes bits off the stream and
/// changes nothing a decoder hands back: the elements exactly, each narrow
/// presentation its fold, and no early substream past the eight matrices
/// FFmpeg reads. Against the same programme with it turned off, which is the
/// stream as it was before.
#[test]
fn decorrelating_the_hierarchy_saves_bits_and_changes_nothing_decoded() {
    let (planes, presentations) = fold_programme();
    let written = |decorrelate: bool| {
        let config = config(planes.len(), 48_000, SampleBits::TwentyFour);
        let mut encoder = Encoder::new(config).expect("twelve elements");
        encoder.set_decorrelation(decorrelate);
        encoder.set_presentations(&presentations);
        let stream = encode(&mut encoder, &planes);
        assert_eq!(encoder.stats().folds_carried, 2, "both intervals fold");
        stream
    };
    let plain = written(false);
    let decorrelated = written(true);
    assert!(
        decorrelated.len() < plain.len(),
        "{} bytes decorrelated against {} without",
        decorrelated.len(),
        plain.len()
    );
    assert_folds_hold(&decorrelated, &planes, &presentations);

    let mut at = 0;
    let mut count = 0;
    while at < decorrelated.len() {
        let unit = hz_mlp::reader::access_unit(&decorrelated[at..], 4, false).expect("a unit");
        for (which, substream) in unit.substreams.iter().enumerate().take(3) {
            if substream.restart.is_some() {
                assert!(
                    substream.matrices.len() <= hz_mlp::matrix::MAX_MATRICES,
                    "substream {which} declares {} matrices",
                    substream.matrices.len()
                );
            }
        }
        at += unit.bytes;
        count += 1;
    }
    assert!(count > 0);
}

/// A programme of twelve elements and its 2.0, 5.1 and 7.1 folds: beds
/// routed and objects panned, over two restart intervals of noise and tones
/// at twenty bits in a twenty-four bit container.
fn fold_programme() -> (Vec<Vec<i32>>, Vec<hz_mlp::hierarchy::Presentation>) {
    use hz_mlp::hierarchy::Presentation;
    const ELEMENTS: usize = 12;
    // One restart interval and a little of the next.
    let frames = 40 * 130;
    let mut planes: Vec<Vec<i32>> = (0..ELEMENTS)
        .map(|element| match element {
            0 => tone(frames, 480.0, 300_000.0),
            8 | 9 => vec![0; frames],
            11 => tone(frames, 23.0, 3_000_000.0),
            _ => noise(0x9e37_79b9 + element as u64, frames, 17),
        })
        .collect();
    // Twenty-bit elements in a twenty-four bit container, as a programme's are.
    for plane in &mut planes {
        for sample in plane {
            *sample &= !0xf;
        }
    }

    let row = |gains: &[(usize, f64)]| {
        let mut out = vec![0.0; ELEMENTS];
        for (element, gain) in gains {
            out[*element] = *gain;
        }
        out
    };
    let mix = |rows: &[(&[f64], f64)]| {
        let mut out = vec![0.0; ELEMENTS];
        for (row, gain) in rows {
            for (slot, value) in out.iter_mut().zip(*row) {
                *slot += gain * value;
            }
        }
        out
    };
    // 7.1 over the twelve elements: beds routed, objects panned.
    let seven = vec![
        row(&[(1, 1.0), (3, 0.7), (11, 0.5), (10, 0.2)]),
        row(&[(2, 1.0), (3, 0.7), (11, 0.5), (10, 0.2)]),
        row(&[(3, 0.7), (11, 0.7)]),
        row(&[(0, 1.0)]),
        row(&[(5, 0.87), (7, 0.5), (10, 0.4)]),
        row(&[(6, 0.87), (7, 0.5), (10, 0.4)]),
        row(&[(4, 0.84), (7, 0.3), (9, 0.5)]),
        row(&[(4, 0.84), (7, 0.3), (8, 0.5)]),
    ];
    let five = vec![
        seven[0].clone(),
        seven[1].clone(),
        seven[2].clone(),
        seven[3].clone(),
        mix(&[(&seven[4], 1.0), (&seven[6], 0.87)]),
        mix(&[(&seven[5], 1.0), (&seven[7], 0.87)]),
    ];
    let two = vec![
        mix(&[(&five[0], 0.5), (&five[2], 0.354), (&five[4], 0.5)]),
        mix(&[(&five[1], 0.5), (&five[2], 0.354), (&five[5], 0.5)]),
    ];
    let elements: Vec<Vec<f64>> = (0..ELEMENTS)
        .map(|element| row(&[(element, 1.0)]))
        .collect();
    let presentations: Vec<Presentation> = [(2usize, two), (6, five), (8, seven), (12, elements)]
        .into_iter()
        .map(|(channels, rows)| Presentation { channels, rows })
        .collect();

    (planes, presentations)
}

/// The elements come back exactly, and each narrow presentation is its fold.
fn assert_folds_hold(
    stream: &[u8],
    planes: &[Vec<i32>],
    presentations: &[hz_mlp::hierarchy::Presentation],
) {
    let elements = planes.len();
    let frames = planes[0].len();
    // The elements, exactly, and in order.
    let full = decoder::decode(&stream, 3).expect("the elements decode");
    assert_eq!(full.channels, elements);
    assert_eq!(full.frames(), frames);
    for (frame, got) in full.samples.chunks_exact(elements).enumerate() {
        for (element, sample) in got.iter().enumerate() {
            assert_eq!(
                *sample, planes[element][frame],
                "frame {frame}, element {element}"
            );
        }
    }

    // And each narrow presentation is its fold: the same signal to within
    // the coefficients' own grid, at the same level.
    for (which, presentation) in presentations[..3].iter().enumerate() {
        let decoded =
            decoder::decode(&stream, which).unwrap_or_else(|e| panic!("presentation {which}: {e}"));
        assert_eq!(decoded.channels, presentation.channels);
        for (channel, fold) in presentation.rows.iter().enumerate() {
            let wanted: Vec<f64> = (0..frames)
                .map(|frame| {
                    fold.iter()
                        .enumerate()
                        .map(|(element, gain)| gain * f64::from(planes[element][frame]))
                        .sum()
                })
                .collect();
            let got: Vec<f64> = (0..frames)
                .map(|frame| f64::from(decoded.frame(frame)[channel]))
                .collect();
            let dot: f64 = wanted.iter().zip(&got).map(|(a, b)| a * b).sum();
            let ww: f64 = wanted.iter().map(|a| a * a).sum();
            let gg: f64 = got.iter().map(|b| b * b).sum();
            let correlation = dot / (ww * gg).sqrt().max(1e-9);
            let level = 10.0 * (gg / ww.max(1e-9)).log10();
            assert!(
                correlation > 0.9995,
                "presentation {which}, channel {channel}: correlation {correlation:.5}"
            );
            assert!(
                level.abs() < 0.2,
                "presentation {which}, channel {channel}: {level:+.2} dB off its fold"
            );
        }
    }
}

/// A fold the codec's domain cannot hold is not written, and the encoder says
/// so rather than leaving it to `HZ_FOLD`: the interval carries the leading
/// elements instead, still lossless, and is counted as refused over the
/// domain rather than carried.
#[test]
fn a_fold_the_domain_cannot_hold_is_counted_and_not_written() {
    use hz_mlp::hierarchy::Presentation;
    const ELEMENTS: usize = 12;
    let frames = 40 * 130;
    // Every element at the full twenty-four bits: no dead bits to lend the
    // cascade the headroom it stores its sums in.
    let planes: Vec<Vec<i32>> = (0..ELEMENTS)
        .map(|element| noise(0xf011_5ca1 + element as u64, frames, 24))
        .collect();
    let row = |gains: &[(usize, f64)]| {
        let mut out = vec![0.0; ELEMENTS];
        for (element, gain) in gains {
            out[*element] = *gain;
        }
        out
    };
    let seven: Vec<Vec<f64>> = (0..8)
        .map(|channel| row(&[(channel, 1.0), (8 + channel % 4, 0.5)]))
        .collect();
    let five: Vec<Vec<f64>> = seven[..6].to_vec();
    let two = vec![
        row(&[(0, 0.7), (2, 0.5), (4, 0.7), (8, 0.35)]),
        row(&[(1, 0.7), (2, 0.5), (5, 0.7), (9, 0.35)]),
    ];
    let elements: Vec<Vec<f64>> = (0..ELEMENTS)
        .map(|element| row(&[(element, 1.0)]))
        .collect();
    let presentations: Vec<Presentation> = [(2usize, two), (6, five), (8, seven), (12, elements)]
        .into_iter()
        .map(|(channels, rows)| Presentation { channels, rows })
        .collect();

    let config = config(ELEMENTS, 48_000, SampleBits::TwentyFour);
    let mut encoder = Encoder::new(config).expect("twelve elements");
    encoder.set_presentations(&presentations);
    let stream = encode(&mut encoder, &planes);

    let stats = encoder.stats();
    assert_eq!(stats.folds_asked, 2, "two restart intervals, both asked");
    assert_eq!(stats.folds_carried, 0, "no interval carries a fold");
    assert_eq!(
        stats.folds_over_the_domain, stats.folds_asked,
        "and each was refused for the domain"
    );
    let full = decoder::decode(&stream, 3).expect("the elements decode");
    for (frame, got) in full.samples.chunks_exact(ELEMENTS).enumerate() {
        for (element, sample) in got.iter().enumerate() {
            assert_eq!(
                *sample, planes[element][frame],
                "frame {frame}, element {element}"
            );
        }
    }
}

/// A programme-like sixteen-channel input: tones at different pitches, a
/// little noise under them, and two objects that are nothing but noise.
fn programme_16(frames: usize) -> Vec<Vec<i32>> {
    (0..16)
        .map(|channel| match channel {
            14 | 15 => noise(0x4141 + channel as u64, frames, 22),
            _ => {
                let tonal = tone(frames, 20.0 + 7.3 * channel as f64, 2_000_000.0);
                let hiss = noise(0x1000 + channel as u64, frames, 14);
                tonal.iter().zip(&hiss).map(|(a, b)| a + b).collect()
            }
        })
        .collect()
}

/// The real-time effort writes streams the reference decoder reads back
/// exactly, at the shapes the transcoder uses and across restart intervals.
#[test]
fn realtime_effort_round_trips() {
    // Three restart intervals and a ragged tail.
    let frames = 3 * 128 * 40 + 17;
    round_trip(
        config(16, 48_000, SampleBits::TwentyFour),
        &programme_16(frames),
        Effort::Realtime,
    );
    round_trip(
        config(8, 48_000, SampleBits::TwentyFour),
        &programme_16(frames)[..8].to_vec(),
        Effort::Realtime,
    );
    let planes = vec![
        tone(frames, 4.7, 7_000_000.0),
        noise(0x9999_aaaa, frames, 21),
    ];
    round_trip(
        config(2, 48_000, SampleBits::TwentyFour),
        &planes,
        Effort::Realtime,
    );
}

/// White noise on all sixteen channels, which prediction cannot help at all.
///
/// Sixteen channels of 24-bit audio are 18.4 Mbit/s uncompressed, above the
/// 18.0 Mbit/s the stream declares, so no lossless coder fits full-scale noise
/// on every channel; that case is only held to the slot. Noise a couple of bits
/// below full scale is the worst input the cap leaves room for, and there the
/// stream averages under the declared peak and the arrival schedule starves nothing.
/// Both stay inside a MAT slot's 61440 bytes per 20 ms (twenty-four units of
/// forty samples), and both decode back exactly.
#[test]
fn realtime_effort_on_white_noise_fits_the_rate_cap_and_the_slot() {
    let frames = 2 * 128 * 40;
    for (bits, fits_the_cap) in [(22, true), (23, true), (24, false)] {
        let planes: Vec<Vec<i32>> = (0..16)
            // Odd and far apart, so no two channels share a sequence.
            .map(|channel| {
                noise(
                    (channel as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                    frames,
                    bits,
                )
            })
            .collect();
        let config = config(16, 48_000, SampleBits::TwentyFour);
        let mut encoder = Encoder::new(config).unwrap();
        encoder.set_effort(Effort::Realtime);
        let stream = encode(&mut encoder, &planes);
        check(&encoder, &stream, &planes, config);
        let stats = encoder.stats();
        let cap = encoder.declared_peak_bitrate();
        let rate = (stats.peak_unit_bytes as f64) * 8.0 / (40.0 / 48_000.0);
        println!(
            "{bits}-bit noise: largest unit {} bytes = {rate:.0} bit/s against {cap}, starved {}",
            stats.peak_unit_bytes, stats.starved
        );
        assert!(
            stats.peak_unit_bytes * 24 <= 61_440,
            "{bits}-bit noise: twenty-four of the largest units overfill a MAT slot"
        );
        if fits_the_cap {
            assert_eq!(
                stats.starved, 0,
                "{bits}-bit noise: a unit could not be placed"
            );
            let mean = stats.total_bytes as f64 * 8.0 / (frames as f64 / 48_000.0);
            assert!(
                mean <= cap as f64,
                "{bits}-bit noise: the stream averages {mean:.0} bit/s"
            );
        }
    }
}

/// Loud, full-bandwidth 24-bit material (tones near full scale with hiss, and
/// two noisy channels) stays inside the declared rate: the cap is a limit on
/// content that is nearly incompressible, not on loud music.
#[test]
fn realtime_effort_keeps_loud_24_bit_content_under_the_cap() {
    let frames = 4 * 128 * 40;
    let planes: Vec<Vec<i32>> = (0..16)
        .map(|channel| match channel {
            14 | 15 => noise(
                (channel as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                frames,
                20,
            ),
            _ => {
                let tonal = tone(frames, 17.0 + 5.1 * channel as f64, 7_500_000.0);
                let hiss = noise(
                    (channel as u64 + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9),
                    frames,
                    16,
                );
                tonal.iter().zip(&hiss).map(|(a, b)| a + b).collect()
            }
        })
        .collect();
    let config = config(16, 48_000, SampleBits::TwentyFour);
    let mut encoder = Encoder::new(config).unwrap();
    encoder.set_effort(Effort::Realtime);
    let stream = encode(&mut encoder, &planes);
    check(&encoder, &stream, &planes, config);
    let stats = encoder.stats();
    let cap = encoder.declared_peak_bitrate();
    let mean = stats.total_bytes as f64 * 8.0 / (frames as f64 / 48_000.0);
    println!(
        "loud 24-bit content: mean {mean:.0} bit/s against {cap}, starved {}",
        stats.starved
    );
    assert_eq!(stats.starved, 0);
    assert!(mean <= cap as f64);
    assert!(stats.peak_unit_bytes * 24 <= 61_440);
}

/// Folds stated by a caller over sixteen elements (a 2.0 and a 5.1 folded from
/// an eight-element bed, the 7.1 the bed itself) hold in the real-time effort:
/// the sixteen elements come back exactly and every narrower presentation is
/// its fold.
#[test]
fn realtime_effort_carries_real_folds_for_sixteen_elements() {
    use hz_mlp::hierarchy::Presentation;
    const ELEMENTS: usize = 16;
    let frames = 3 * 128 * 40;
    let planes: Vec<Vec<i32>> = (0..ELEMENTS)
        .map(|channel| {
            let tonal = tone(frames, 23.0 + 6.1 * channel as f64, 1_500_000.0);
            let hiss = noise(
                (channel as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                frames,
                12,
            );
            tonal.iter().zip(&hiss).map(|(a, b)| a + b).collect()
        })
        .collect();
    let row = |gains: &[(usize, f64)]| {
        let mut out = vec![0.0; ELEMENTS];
        for &(element, gain) in gains {
            out[element] = gain;
        }
        out
    };
    let seven: Vec<Vec<f64>> = (0..8).map(|c| row(&[(c, 1.0)])).collect();
    let five = vec![
        row(&[(0, 1.0)]),
        row(&[(1, 1.0)]),
        row(&[(2, 1.0)]),
        row(&[(3, 1.0)]),
        row(&[(4, 0.707), (6, 0.707)]),
        row(&[(5, 0.707), (7, 0.707)]),
    ];
    let two = vec![
        row(&[(0, 0.5), (2, 0.354), (4, 0.354), (6, 0.354)]),
        row(&[(1, 0.5), (2, 0.354), (5, 0.354), (7, 0.354)]),
    ];
    let elements: Vec<Vec<f64>> = (0..ELEMENTS).map(|c| row(&[(c, 1.0)])).collect();
    let presentations: Vec<Presentation> =
        [(2usize, two), (6, five), (8, seven), (ELEMENTS, elements)]
            .into_iter()
            .map(|(channels, rows)| Presentation { channels, rows })
            .collect();

    let config = config(ELEMENTS, 48_000, SampleBits::TwentyFour);
    let mut encoder = Encoder::new(config).unwrap();
    encoder.set_effort(Effort::Realtime);
    encoder.set_presentations(&presentations);
    let stream = encode(&mut encoder, &planes);
    let stats = encoder.stats();
    assert_eq!(
        stats.folds_carried, stats.folds_asked,
        "every interval carries its folds"
    );
    assert!(stats.folds_carried > 0);
    assert_folds_hold(&stream, &planes, &presentations);
}

/// Feeding planes in lengths that are not a whole number of units writes
/// exactly the stream feeding the units one by one does, ragged end included.
#[test]
fn planes_in_any_length_write_the_same_stream_as_units() {
    let frames = 2 * 128 * 40 + 133;
    let planes = programme_16(frames);
    let config = config(16, 48_000, SampleBits::TwentyFour);
    let mut by_units = Encoder::new(config).unwrap();
    by_units.set_effort(Effort::Realtime);
    let wanted = encode(&mut by_units, &planes);

    for chunk in [512usize, 1, 39, 40, 41, 1000] {
        let mut encoder = Encoder::new(config).unwrap();
        encoder.set_effort(Effort::Realtime);
        assert!(
            encoder.push_planes(&[], 0).is_empty(),
            "no frames, no bytes"
        );
        let mut stream = Vec::new();
        let mut at = 0;
        while at < frames {
            let n = chunk.min(frames - at);
            let laid: Vec<i32> = planes
                .iter()
                .flat_map(|p| p[at..at + n].iter().copied())
                .collect();
            stream.extend_from_slice(&encoder.push_planes(&laid, n));
            at += n;
        }
        stream.extend_from_slice(&encoder.finish_carried());
        assert_eq!(stream, wanted, "chunks of {chunk}");
    }
}
