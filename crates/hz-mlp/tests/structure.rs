//! Read back what the encoder writes, without a decoder.
//!
//! The real acceptance test for a lossless encoder is `decode(encode(x)) == x`
//! through somebody else's decoder, and that lives in `cargo xtask mlp`
//! because it needs one installed. This is what can be checked without one:
//! that the stream describes itself consistently — the lengths agree with the
//! bytes, the sync words are where the headers say they are, and every
//! checksum verifies.
//!
//! Which is worth having on its own. Every structural mistake made while
//! writing this encoder — a length counted in bytes where the format counts
//! words, a checksum over the wrong range — would have been caught here, and
//! caught with an error naming the field rather than "the decoder produced
//! silence".

use hz_mlp::{Config, Encoder, SampleBits, crc};

/// One access unit, located and checked.
struct Unit {
    total: usize,
    major_sync: bool,
    substream_end: usize,
}

/// Walk one access unit, failing with a description of what disagreed.
fn parse(stream: &[u8]) -> Result<Unit, String> {
    if stream.len() < 4 {
        return Err(format!("{} bytes is not an access unit", stream.len()));
    }
    let header = u16::from_be_bytes([stream[0], stream[1]]);
    let total = usize::from(header & 0x0fff) * 2;
    if total > stream.len() {
        return Err(format!(
            "header says {total} bytes, buffer holds {}",
            stream.len()
        ));
    }

    // The parity nibble covers the timing, the length and the substream
    // headers. Recomputing it here is the only check that the header agrees
    // with what follows it.
    let input_timing = u16::from_be_bytes([stream[2], stream[3]]);
    let major_sync = stream.len() >= 8 && stream[4..7] == [0xf8, 0x72, 0x6f];
    let substream_header_at = if major_sync { 4 + 28 } else { 4 };
    if substream_header_at + 2 > total {
        return Err("no room for a substream header".into());
    }

    let mut parity = input_timing ^ (header & 0x0fff);
    parity ^= u16::from(stream[substream_header_at]);
    parity ^= u16::from(stream[substream_header_at + 1]);
    parity ^= parity >> 8;
    parity ^= parity >> 4;
    let expected = (parity & 0xf) ^ 0xf;
    if header >> 12 != expected {
        return Err(format!(
            "access unit parity nibble is {:x}, should be {expected:x}",
            header >> 12
        ));
    }

    if major_sync {
        let sync = &stream[4..4 + 28];
        let checksum = crc::major_sync(&sync[..26]);
        if checksum != [sync[26], sync[27]] {
            return Err(format!(
                "major sync checksum is {:02x?}, should be {checksum:02x?}",
                &sync[26..28]
            ));
        }
    }

    let substream_header =
        u16::from_be_bytes([stream[substream_header_at], stream[substream_header_at + 1]]);
    if substream_header & (1 << 13) == 0 {
        return Err("the substream does not declare a checksum".into());
    }
    if (substream_header >> 14) & 1 == u16::from(major_sync) {
        return Err("nonrestart_substr disagrees with the major sync".into());
    }

    let payload_at = substream_header_at + 2;
    let end = usize::from(substream_header & 0x0fff) * 2;
    if payload_at + end != total {
        return Err(format!(
            "substream ends at {} but the unit ends at {total}",
            payload_at + end
        ));
    }

    let payload = &stream[payload_at..payload_at + end];
    if payload.len() < 2 {
        return Err("a substream is at least its parity and checksum".into());
    }
    let body = &payload[..payload.len() - 2];
    let parity = crc::parity(body) ^ crc::PARITY_XOR;
    if parity != payload[payload.len() - 2] {
        return Err(format!(
            "substream parity is {:02x}, should be {parity:02x}",
            payload[payload.len() - 2]
        ));
    }
    let checksum = crc::substream(body);
    if checksum != payload[payload.len() - 1] {
        return Err(format!(
            "substream checksum is {:02x}, should be {checksum:02x}",
            payload[payload.len() - 1]
        ));
    }

    Ok(Unit {
        total,
        major_sync,
        substream_end: end,
    })
}

fn walk(stream: &[u8]) -> Vec<Unit> {
    let mut units = Vec::new();
    let mut at = 0;
    while at < stream.len() {
        match parse(&stream[at..]) {
            Ok(unit) => {
                at += unit.total;
                units.push(unit);
            }
            Err(why) => panic!("access unit {} at byte {at}: {why}", units.len()),
        }
    }
    units
}

fn encode(sample_rate: u32, channels: usize, bits: SampleBits, frames: usize) -> Vec<u8> {
    let mut encoder = Encoder::new(Config {
        sample_rate,
        channels,
        bits,
    })
    .expect("a shape the encoder writes");

    // Something with structure in it, so a writer that drops samples shows up
    // in the length rather than in silence.
    let peak = match bits {
        SampleBits::Sixteen => 32_000i64,
        SampleBits::TwentyFour => 8_000_000,
    };
    let samples: Vec<i32> = (0..frames * channels)
        .map(|n| {
            let phase = n as f64 / 37.0;
            (phase.sin() * peak as f64) as i32
        })
        .collect();

    let unit = encoder.frame_size() * channels;
    let mut stream = Vec::new();
    let mut at = 0;
    while at + unit <= samples.len() {
        stream.extend_from_slice(&encoder.push(&samples[at..at + unit]));
        at += unit;
    }
    stream.extend_from_slice(&encoder.finish(&samples[at..]));
    stream
}

#[test]
fn every_shape_the_encoder_accepts_describes_itself_consistently() {
    for rate in [44_100u32, 48_000, 88_200, 96_000, 176_400, 192_000] {
        for channels in [1usize, 2] {
            for bits in [SampleBits::Sixteen, SampleBits::TwentyFour] {
                // Long enough that several restart intervals go by at every
                // rate, since the unit length changes with the rate.
                // Enough for two restarts at every rate: the frame size grows
                // with the rate, and a restart is one per 128 of them.
                let stream = encode(rate, channels, bits, 45_000);
                let units = walk(&stream);
                assert!(
                    !units.is_empty(),
                    "{rate} Hz, {channels} ch, {bits:?}: nothing was written"
                );
                assert!(
                    units[0].major_sync,
                    "{rate} Hz: a stream has to begin with a restart"
                );
                // Restarts are periodic, and the period is the encoder's
                // business — but they have to be regular, because a decoder
                // joining a stream mid-way waits for one.
                let restarts: Vec<usize> = units
                    .iter()
                    .enumerate()
                    .filter(|(_, unit)| unit.major_sync)
                    .map(|(index, _)| index)
                    .collect();
                let gaps: Vec<usize> = restarts.windows(2).map(|w| w[1] - w[0]).collect();
                assert!(
                    gaps.windows(2).all(|w| w[0] == w[1]),
                    "{rate} Hz: restarts at {restarts:?} are not evenly spaced"
                );
                assert!(
                    restarts.len() > 1,
                    "{rate} Hz: {} units and only one restart",
                    units.len()
                );
                assert!(
                    gaps.first().is_none_or(|gap| *gap <= 128),
                    "{rate} Hz: restarts {gaps:?} apart, and a decoder refuses more than 128"
                );
                assert!(
                    units.iter().all(|u| u.substream_end.is_multiple_of(2)),
                    "{rate} Hz: a substream is measured in 16-bit words"
                );
            }
        }
    }
}

/// A stream whose length is not a whole number of units still ends cleanly,
/// and the last unit is a unit like any other.
#[test]
fn a_ragged_length_still_ends_on_a_unit() {
    let stream = encode(48_000, 2, SampleBits::TwentyFour, 40 * 7 + 13);
    let units = walk(&stream);
    assert_eq!(units.len(), 8, "seven whole units and a short one");
}

/// A restart is what a decoder joining a stream part-way waits for, so they
/// have to keep coming — and the format says how often at the least: a
/// decoder refuses a gap of more than 128 access units by name.
#[test]
fn restarts_keep_coming_at_least_as_often_as_the_format_demands() {
    let stream = encode(48_000, 2, SampleBits::TwentyFour, 40 * 600);
    let units = walk(&stream);
    let restarts: Vec<usize> = units
        .iter()
        .enumerate()
        .filter(|(_, unit)| unit.major_sync)
        .map(|(index, _)| index)
        .collect();
    assert!(
        restarts.len() > 2,
        "{} restarts in 600 units",
        restarts.len()
    );
    let widest = restarts.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0);
    assert!(
        widest <= 128,
        "a gap of {widest} units, which a decoder refuses"
    );
}

/// A length that lands exactly on a boundary must not gain a unit holding
/// nothing, which a decoder reports as a truncated stream.
#[test]
fn an_exact_length_gains_no_empty_unit() {
    let stream = encode(48_000, 2, SampleBits::TwentyFour, 40 * 7);
    assert_eq!(walk(&stream).len(), 7);
}

/// Dialogue normalisation 31 is "no normalisation"; 0 would be a 31 dB cut at
/// a decoder that applies it. Every restart of every shape says 31, in the
/// two-, six- and eight-channel fields alike.
#[test]
fn every_major_sync_leaves_the_level_alone() {
    for channels in [2usize, 6, 8] {
        let stream = encode(48_000, channels, SampleBits::TwentyFour, 40 * 300);
        let units = hz_mlp::reader::stream(&stream).expect("the stream reads back");
        let syncs: Vec<_> = units.iter().filter_map(|u| u.major_sync.as_ref()).collect();
        assert!(syncs.len() > 1, "{channels} ch: more than one restart");
        for sync in syncs {
            assert!(sync.checksum_ok);
            assert_eq!(
                (
                    sync.dialogue_norm_2ch,
                    sync.dialogue_norm_6ch,
                    sync.dialogue_norm_8ch
                ),
                (31, 31, 31),
                "{channels} ch"
            );
        }
    }
}
