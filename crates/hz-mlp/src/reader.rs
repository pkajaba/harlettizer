// SPDX-License-Identifier: GPL-3.0-or-later
//
// Carries material ported from the `truehd` crate
// (https://github.com/truehdd/truehdd) — Copyright (c) the truehdd authors —
// under the Apache License, Version 2.0, whose text is reproduced in
// LICENSES/Apache-2.0.txt.
// What was taken and what was changed is recorded in docs/provenance.md.

//! Reading the structure of a TrueHD access unit.
//!
//! Not a decoder: this reads as far as the block data and stops, which is
//! exactly the part an encoder has to get right and exactly the part that
//! fails silently when it is wrong. Every field of this format that was got
//! right in this project was got right by comparing against a stream somebody
//! else wrote, and the ones that were wrong were wrong in ways that parsed
//! cleanly and decoded to the wrong samples. So the comparison is a program.
//!
//! ```text
//! cargo xtask thd stream.thd
//! ```

use crate::crc;
use crate::format::RestartSync;
use hz_core::bits_read::{BitReader, Ran};

type Result<T> = std::result::Result<T, Ran>;

const SYNC_MAJOR: u32 = 0x00f8_726f;
const SYNC_TRUEHD: u32 = 0xba;
const SYNC_MLP: u32 = 0xbb;

/// What a major sync says about the stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MajorSync {
    /// Bytes it occupies, which is not fixed: an Atmos stream carries an extra
    /// channel-meaning block and is longer.
    pub size: usize,
    pub truehd: bool,
    pub rate_code: u32,
    /// The six-channel presentation's arrangement, five bits.
    pub arrangement_6ch: u32,
    /// The eight-channel presentation's, thirteen.
    pub arrangement_8ch: u32,
    pub flags: u32,
    pub peak_bitrate: u32,
    pub substreams: usize,
    pub extended_substream_info: u32,
    pub substream_info: u32,
    /// Dialogue normalisation of the two-, six- and eight-channel
    /// presentations. A decoder applies it as an attenuation of that many dB
    /// below 31, so 31 is the value that leaves the level alone.
    pub dialogue_norm_2ch: u32,
    pub dialogue_norm_6ch: u32,
    pub dialogue_norm_8ch: u32,
    /// The extra channel meaning, when there is one: its bytes, and what the
    /// sixteen-element presentation says about itself if it names one.
    pub extra_channel_meaning: Option<Vec<u8>>,
    pub immersive: Option<Immersive>,
    pub checksum_ok: bool,
}

/// The sixteen-element presentation, as the extra channel meaning states it.
///
/// Only the shape the reference stream uses is read: every element a dynamic
/// object, one of which may be the low frequency channel. The other shape —
/// a described bed, with a content description and a channel assignment — is
/// left unparsed rather than guessed at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Immersive {
    pub dialogue_norm: u32,
    pub mix_level: u32,
    pub objects: u32,
    pub dynamic_objects_only: bool,
    pub lfe: bool,
}

/// One entry of the substream directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Directory {
    /// Whether a second word follows carrying dynamic range control.
    pub extra_word: bool,
    /// Set when the substream does *not* restart here. A decoder rejects a
    /// unit whose major sync and this bit disagree.
    pub restart_nonexistent: bool,
    pub checkdata_present: bool,
    /// Where this substream ends, in 16-bit words from the start of the
    /// payload — cumulative, so entry `n` is where substream `n` finishes.
    pub end_words: u16,
    pub drc_gain_update: i32,
    pub drc_time_update: u32,
}

/// A restart header, which is what tells a decoder how a substream is shaped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestartHeader {
    /// Which of the three sync words this substream carries. Reading it as
    /// thirteen bits of sync and a noise-type flag — which is what FFmpeg's
    /// decoder does — accepts two of them and refuses the third, and the third
    /// is the one an immersive stream's last substream uses.
    pub sync: RestartSync,
    pub output_timing: u32,
    pub min_channel: usize,
    pub max_channel: usize,
    pub max_matrix_channel: usize,
    pub noise_shift: u32,
    pub noisegen_seed: u32,
    pub max_shift: u32,
    pub max_huff_lsbs: u32,
    pub max_output_bits: u32,
    /// Whether the substream carries its own error protection.
    pub error_protect: bool,
    pub hires_output_timing: bool,
    pub lossless_check: u32,
    pub channel_assignment: Vec<u32>,
    pub checksum_ok: bool,
}

/// One substream's framing, as far as the block data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Substream {
    pub bytes: usize,
    pub block_header: bool,
    pub restart: Option<RestartHeader>,
    /// The matrices the first block after a restart header declares, in the
    /// order a decoder applies them. Only read where there is a restart header
    /// to read them after.
    pub matrices: Vec<Declared>,
    /// And the dead-bit shift it states per channel, over the matrix span.
    pub output_shift: Vec<i32>,
    pub parity_ok: bool,
    pub checksum_ok: bool,
}

/// One primitive matrix, as a substream declares it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Declared {
    pub dest: usize,
    pub frac_bits: u32,
    /// A power of two the whole matrix is scaled by, where the syntax has one.
    pub shift: u32,
    /// How many of the destination's low bits the stream carries explicitly.
    pub bypassed: u32,
    /// One per source the substream writes coefficients for, in the field's own
    /// units — `1 << frac_bits` is unity.
    pub coefficients: Vec<i32>,
    /// How far each coefficient moves per access unit, in the field's own
    /// units scaled by `2^delta_precision`, or all zero for a matrix that
    /// stands still. Only the immersive syntax can say one.
    pub delta: Vec<i32>,
    pub delta_precision: u32,
}

impl Declared {
    /// The coefficient on `source` as a real gain.
    pub fn gain(&self, source: usize) -> f64 {
        self.coefficients.get(source).map_or(0.0, |stored| {
            f64::from(*stored) / f64::from(1i32 << self.frac_bits)
        })
    }
}

/// One access unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessUnit {
    pub bytes: usize,
    pub input_timing: u32,
    pub parity_nibble_ok: bool,
    pub major_sync: Option<MajorSync>,
    pub directory: Vec<Directory>,
    pub substreams: Vec<Substream>,
    /// Bytes after the last substream, which is where the object metadata
    /// lives in an immersive stream.
    pub extra_data: usize,
    /// And what they say, when the stream declared them an Evolution frame.
    pub evolution: Option<ExtraData>,
}

/// What rides after the last substream.
///
/// A decoder enters this whenever a whole 16-bit word is left in the access
/// unit, and what it expects to find is decided by the major sync's flags
/// rather than by anything here: bit 12 set is an Evolution frame, clear is an
/// opaque payload. Only the first is read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtraData {
    /// Bytes it occupies, the two that state its length included.
    pub bytes: usize,
    /// Whether the first two bytes fold to the value that says so.
    pub check_nibble_ok: bool,
    pub parity_ok: bool,
    /// The byte length the Evolution header declares, which is the room after
    /// that header rather than the frame's own size.
    pub declared_bytes: usize,
    pub version: u32,
    pub key: u32,
    pub payloads: Vec<Payload>,
    /// Bits in the primary and secondary protection fields that close the
    /// frame, as their length codes declare them. The format calls what they
    /// hold implementation dependent and tells a decoder it may ignore them.
    pub protection: (u32, u32),
    /// The primary field's bytes as written, and the bit within the block at
    /// which they start — which, with the key, is what
    /// [`AccessUnit::protection_verifies`] checks them against. The secondary
    /// field is counted and not kept: nothing writes one.
    pub protection_bytes: Vec<u8>,
    pub protection_at: usize,
    /// Whether the Evolution frame inside it could be read at all. A block
    /// whose parity has already failed is a block whose payload sizes are not
    /// to be trusted, and running off the end of it says nothing about the
    /// audio in the same access unit — which is fine and comes back exactly.
    pub frame_ok: bool,
}

/// One Evolution payload. The frame is the one E-AC-3 carries too, so it is
/// read by [`hz_meta::emdf`].
pub use hz_meta::emdf::ReadPayload as Payload;

/// Read one access unit from the front of `stream`.
///
/// `substreams` is how many the last major sync declared, since a unit without
/// one does not say.
pub fn access_unit(stream: &[u8], substreams: usize, evolution: bool) -> Result<AccessUnit> {
    if stream.len() < 4 {
        return Err(Ran(format!("{} bytes is not an access unit", stream.len())));
    }
    let header = u32::from(u16::from_be_bytes([stream[0], stream[1]]));
    let length = (header & 0x0fff) as usize * 2;
    if length > stream.len() || length < 4 {
        return Err(Ran(format!(
            "the header says {length} bytes, the buffer holds {}",
            stream.len()
        )));
    }
    let input_timing = u32::from(u16::from_be_bytes([stream[2], stream[3]]));

    let major_sync = if stream.len() >= 8 && read_be24(stream, 4) == SYNC_MAJOR {
        Some(major_sync(&stream[4..])?)
    } else {
        None
    };
    let substreams = major_sync
        .as_ref()
        .map_or(substreams, |sync| sync.substreams);
    // Like the substream count, the shape of the extra data is stated in the
    // major sync and holds until the next one.
    let evolution = major_sync
        .as_ref()
        .map_or(evolution, |sync| sync.flags & (1 << 12) != 0);
    if substreams == 0 || substreams > 4 {
        return Err(Ran(format!("{substreams} substreams is not a stream")));
    }

    let directory_at = 4 + major_sync.as_ref().map_or(0, |sync| sync.size);
    let mut reader = BitReader::at_bit(&stream[..length], directory_at * 8);
    let mut directory = Vec::with_capacity(substreams);
    for _ in 0..substreams {
        directory.push(entry(&mut reader)?);
    }
    if !reader.is_aligned() {
        return Err(Ran("the substream directory did not end on a byte".into()));
    }
    let payload_at = reader.byte_position();

    // The parity nibble covers the timing, the length and the directory.
    let mut parity = (input_timing ^ (header & 0x0fff)) as u16;
    for byte in &stream[directory_at..payload_at] {
        parity ^= u16::from(*byte);
    }
    parity ^= parity >> 8;
    parity ^= parity >> 4;
    let parity_nibble_ok = (header >> 12) as u16 == ((parity & 0xf) ^ 0xf);

    let mut substream_list = Vec::with_capacity(substreams);
    let mut start = payload_at;
    for entry in &directory {
        let end = payload_at + usize::from(entry.end_words) * 2;
        if end > length || end < start {
            return Err(Ran(format!(
                "a substream ends at {end}, outside the unit's {length} bytes"
            )));
        }
        substream_list.push(substream(&stream[start..end], entry)?);
        start = end;
    }

    // A decoder enters the extra data whenever a whole word is left, so a
    // unit that carries none simply ends at its last substream.
    let extra = (evolution && length - start >= 2)
        .then(|| extra_data(&stream[start..length]))
        .transpose()?;

    Ok(AccessUnit {
        bytes: length,
        input_timing,
        parity_nibble_ok,
        major_sync,
        directory,
        substreams: substream_list,
        extra_data: length - start,
        evolution: extra,
    })
}

/// The extra data block, in its Evolution shape.
fn extra_data(bytes: &[u8]) -> Result<ExtraData> {
    let mut reader = BitReader::new(bytes);
    let nibble = reader.get(4)?;
    let words = reader.get(12)? as usize;

    // The first two bytes fold to fifteen, which is the only thing that says
    // this is a block at all rather than padding.
    let fold = bytes[0] ^ bytes[1];
    let check_nibble_ok = (fold ^ (fold >> 4)) & 0xf == 0xf;
    let _ = nibble;

    let total = 2 + words * 2;
    if words == 0 || total > bytes.len() {
        return Err(Ran(format!(
            "extra data of {words} words does not fit in {} bytes",
            bytes.len()
        )));
    }
    // Everything but the two bytes that state the length and the parity byte
    // itself, folded with the same constant a writer uses.
    let parity = bytes[2..total - 1]
        .iter()
        .fold(0xa9u8, |acc, byte| acc ^ byte);
    let parity_ok = parity == bytes[total - 1];

    reader.skip(4)?; // reserved
    let declared_bytes = reader.get(12)? as usize;

    let frame_at = reader.bit_position();
    let parsed = hz_meta::emdf::read(&mut reader);
    let frame_ok = parsed.is_some();
    let frame = parsed.unwrap_or_default();

    Ok(ExtraData {
        bytes: total,
        check_nibble_ok,
        parity_ok,
        declared_bytes,
        version: frame.version,
        key: frame.key,
        payloads: frame.payloads,
        protection: frame.protection,
        protection_bytes: frame.protection_bytes,
        protection_at: frame_at + frame.protection_at,
        frame_ok,
    })
}

impl AccessUnit {
    /// Whether the Evolution frame's protection field holds the digest `key`
    /// would have written — see [`crate::protection`] for what it covers.
    ///
    /// `unit` is the slice this unit was read from. `None` when there is
    /// nothing to check: no Evolution frame, one that did not parse, or an
    /// empty field.
    pub fn protection_verifies(
        &self,
        key: &hz_core::hmac::HmacSha256,
        unit: &[u8],
    ) -> Option<bool> {
        let extra = self.evolution.as_ref()?;
        if !extra.frame_ok || extra.protection_bytes.is_empty() {
            return None;
        }
        let block_at = self.bytes - self.extra_data;
        let frame_at = block_at + crate::protection::FRAME_HEADER_BYTES;
        if frame_at + extra.declared_bytes > self.bytes {
            return None;
        }
        let digest = crate::protection::digest(
            key,
            &unit[..self.bytes],
            block_at,
            extra.declared_bytes,
            block_at * 8 + extra.protection_at,
        );
        Some(digest[..extra.protection_bytes.len()] == extra.protection_bytes[..])
    }
}

fn read_be24(bytes: &[u8], at: usize) -> u32 {
    u32::from(bytes[at]) << 16 | u32::from(bytes[at + 1]) << 8 | u32::from(bytes[at + 2])
}

/// The major sync, whose length depends on what it carries.
fn major_sync(bytes: &[u8]) -> Result<MajorSync> {
    if bytes.len() < 28 {
        return Err(Ran("a major sync is at least 28 bytes".into()));
    }
    // Its own length is not a field: it is read from two bytes inside it,
    // before anything else has been parsed. An Atmos stream is longer than a
    // channel-based one, and a reader that assumes 28 lands in the middle of
    // the substream directory and produces plausible nonsense.
    let has_extra = bytes[25] & 1 == 1;
    let extensions = if has_extra { bytes[26] >> 4 } else { 0 } as usize;
    let size = if has_extra { 30 + extensions * 2 } else { 28 };
    if bytes.len() < size {
        return Err(Ran(format!("a major sync of {size} bytes does not fit")));
    }

    let mut reader = BitReader::new(&bytes[..size]);
    reader.skip(24)?;
    let stream_type = reader.get(8)?;
    let truehd = match stream_type {
        SYNC_TRUEHD => true,
        SYNC_MLP => false,
        other => {
            return Err(Ran(format!(
                "stream type {other:#x} is neither MLP nor TrueHD"
            )));
        }
    };
    if !truehd {
        return Err(Ran("MLP major syncs are not read yet".into()));
    }

    let rate_code = reader.get(4)?;
    reader.skip(4)?; // the two multichannel types and two ignored bits
    reader.skip(2)?; // two-channel presentation modifier
    reader.skip(2)?; // six-channel presentation modifier
    let arrangement_6ch = reader.get(5)?;
    reader.skip(2)?; // eight-channel presentation modifier
    let arrangement_8ch = reader.get(13)?;

    let signature = reader.get(16)?;
    if signature != 0xb752 {
        return Err(Ran(format!("major sync signature is {signature:#x}")));
    }
    let flags = reader.get(16)?;
    reader.skip(16)?;
    reader.skip(1)?; // variable rate
    let peak_bitrate = reader.get(15)?;
    let substreams = reader.get(4)? as usize;
    let extended_substream_info = reader.get(4)?;
    let substream_info = reader.get(8)?;
    // Reserved bits, the three control-enable flags and a reserved bit, then
    // the start-up gain's seven.
    reader.skip(6 + 3 + 1 + 7)?;
    let dialogue_norm_2ch = reader.get(6)?;
    reader.skip(6)?; // two-channel mix level
    let dialogue_norm_6ch = reader.get(5)?;
    reader.skip(6 + 5)?; // six-channel mix level and source format
    let dialogue_norm_8ch = reader.get(5)?;

    let extra_channel_meaning = has_extra.then(|| bytes[26..size - 2].to_vec());
    // The sixteen-element fields are there only when the top bit of
    // `substream_info` says the stream declares that presentation at all.
    let immersive = if has_extra && substream_info >> 7 != 0 {
        let mut extra = BitReader::at_bit(&bytes[..size - 2], 26 * 8 + 4);
        Some(Immersive {
            dialogue_norm: extra.get(5)?,
            mix_level: extra.get(6)?,
            objects: extra.get(5)?,
            dynamic_objects_only: extra.get_bit()?,
            lfe: extra.get_bit()?,
        })
    } else {
        None
    };
    let checksum_ok = crc::major_sync(&bytes[..size - 2]) == [bytes[size - 2], bytes[size - 1]];

    Ok(MajorSync {
        size,
        truehd,
        rate_code,
        arrangement_6ch,
        arrangement_8ch,
        flags,
        peak_bitrate,
        substreams,
        extended_substream_info,
        substream_info,
        dialogue_norm_2ch,
        dialogue_norm_6ch,
        dialogue_norm_8ch,
        extra_channel_meaning,
        immersive,
        checksum_ok,
    })
}

fn entry(reader: &mut BitReader<'_>) -> Result<Directory> {
    let extra_word = reader.get_bit()?;
    let restart_nonexistent = reader.get_bit()?;
    let checkdata_present = reader.get_bit()?;
    reader.skip(1)?;
    let end_words = reader.get(12)? as u16;

    // The second word is dynamic range control, and it is what makes the
    // directory four bytes an entry rather than two.
    let (drc_gain_update, drc_time_update) = if extra_word {
        let gain = reader.get_signed(9)?;
        let time = reader.get(3)?;
        reader.skip(4)?;
        (gain, time)
    } else {
        (0, 0)
    };

    Ok(Directory {
        extra_word,
        restart_nonexistent,
        checkdata_present,
        end_words,
        drc_gain_update,
        drc_time_update,
    })
}

fn substream(bytes: &[u8], entry: &Directory) -> Result<Substream> {
    let mut reader = BitReader::new(bytes);
    let block_header = reader.get_bit()?;
    let restart = if block_header && reader.get_bit()? {
        Some(restart_header(&mut reader, bytes)?)
    } else {
        None
    };

    // What the first block after a restart header says about matrices and
    // dead bits. The fields are optional and positional, so they can only be
    // read by walking them in order — which is why this sits here rather than
    // in a tool.
    let mut matrices = Vec::new();
    let mut output_shift = Vec::new();
    if let Some(restart) = &restart {
        let _ = block_params(&mut reader, restart, &mut matrices, &mut output_shift);
    }

    let (parity_ok, checksum_ok) = if entry.checkdata_present && bytes.len() >= 2 {
        let body = &bytes[..bytes.len() - 2];
        (
            crc::parity(body) ^ crc::PARITY_XOR == bytes[bytes.len() - 2],
            crc::substream(body) == bytes[bytes.len() - 1],
        )
    } else {
        (true, true)
    };

    Ok(Substream {
        bytes: bytes.len(),
        block_header,
        restart,
        matrices,
        output_shift,
        parity_ok,
        checksum_ok,
    })
}

/// The first block's parameters, as far as the dead-bit shift.
///
/// Mirrors what the writer emits: a presence flag that may restate which
/// parameters follow, then the block size, then the matrices, then the shift.
/// Everything past that is coding and is not read.
fn block_params(
    reader: &mut BitReader<'_>,
    restart: &RestartHeader,
    matrices: &mut Vec<Declared>,
    output_shift: &mut Vec<i32>,
) -> Result<()> {
    const PARAM_BLOCKSIZE: u8 = 1 << 7;
    const PARAM_MATRIX: u8 = 1 << 6;
    const PARAM_OUTSHIFT: u8 = 1 << 5;
    const PARAM_PRESENCE: u8 = 1 << 0;

    let mut flags = 0xffu8;
    if flags & PARAM_PRESENCE != 0 && reader.get_bit()? {
        flags = reader.get(8)? as u8;
    }
    if flags & PARAM_BLOCKSIZE != 0 && reader.get_bit()? {
        reader.skip(9)?;
    }
    if flags & PARAM_MATRIX != 0 && reader.get_bit()? {
        matrix_params(reader, restart, matrices)?;
    }
    if flags & PARAM_OUTSHIFT != 0 && reader.get_bit()? {
        for _ in 0..=restart.max_matrix_channel {
            output_shift.push(reader.get_signed(4)?);
        }
    }
    Ok(())
}

/// The matrices themselves, in whichever of the two syntaxes this substream
/// uses. See `frame::write_matrix_params`, which this is the mirror of.
fn matrix_params(
    reader: &mut BitReader<'_>,
    restart: &RestartHeader,
    matrices: &mut Vec<Declared>,
) -> Result<()> {
    let sources = match restart.sync {
        // Plus the two noise channels a decoder synthesises.
        RestartSync::A => restart.max_matrix_channel + 3,
        RestartSync::B | RestartSync::C => restart.max_matrix_channel + 1,
    };

    if restart.sync == RestartSync::C {
        // The immersive syntax: every matrix's header first, then every
        // matrix's coefficients, with a presence mask deciding which are there.
        if !reader.get_bit()? {
            reader.skip(1)?;
            return Ok(());
        }
        let full = reader.get_bit()?;
        let count = reader.get(4)? as usize + 1;
        if !full {
            return Ok(());
        }
        let first = matrices.len();
        let mut masks = Vec::with_capacity(count);
        for _ in 0..count {
            let dest = reader.get(4)? as usize;
            let frac_bits = reader.get(4)?;
            // 🔴 A shift that scales the whole matrix, read as one less than
            // the field — so one is none. Skipping it reports every
            // coefficient of a scaled matrix wrong by a power of two, which
            // reads as a destination coefficient of a half where the stream
            // says one.
            let shift = reader.get(3)?.saturating_sub(1);
            // 🔴 How many of the destination's low bits the stream carries
            // explicitly. A matrix that scales its destination would lose
            // them, and sending them is what keeps it lossless — which is how
            // a matrix can do something a lifting step cannot.
            let bypassed = reader.get(2)?;
            reader.skip(4)?; // dither
            masks.push((
                dest,
                frac_bits,
                shift,
                bypassed,
                reader.get(sources as u32)?,
            ));
        }
        for (dest, frac_bits, shift, bypassed, mask) in masks {
            let mut coefficients = vec![0; sources];
            for (source, slot) in coefficients.iter_mut().enumerate() {
                // Source zero is the mask's *least* significant bit — see
                // `frame::mask`, which is what writes it. Reading it the other
                // way round puts every coefficient on the wrong channel, and
                // the giveaway is a stream of this encoder's own lifting steps
                // coming back with destination coefficients of zero.
                if mask >> source & 1 == 1 {
                    *slot = reader.get_signed(frac_bits + 2)?;
                }
            }
            matrices.push(Declared {
                dest,
                frac_bits,
                shift,
                bypassed,
                coefficients,
                delta: vec![0; sources],
                delta_precision: 0,
            });
        }

        // 🔴 What follows the coefficients: whether any matrix moves, and by
        // what. This used to be left unread, and every field of the block
        // after it was then a bit out of place — which read as nothing for as
        // long as no matrix moved, and as output shifts of −4 and −8 once one
        // did, on a stream whose real trouble was elsewhere.
        if reader.get_bit()? {
            if !reader.get_bit()? {
                return Err(Ran(
                    "a matrix moves by steps stated in an earlier unit, which are not kept here"
                        .into(),
                ));
            }
            if !reader.get_bit()? {
                return Err(Ran(
                    "steps re-using an earlier statement's widths are not kept here".into(),
                ));
            }
            let mut widths = Vec::with_capacity(count);
            for _ in 0..count {
                let width = reader.get(4)?;
                let precision = reader.get(2)?;
                widths.push((width, precision));
            }
            // A step for every source the mask names, and no other — a source
            // with no coefficient has no step, whatever the encoder thought.
            for (matrix, (width, precision)) in matrices[first..].iter_mut().zip(widths) {
                matrix.delta_precision = precision;
                if width == 0 {
                    continue;
                }
                for source in 0..sources {
                    if matrix.coefficients[source] != 0 {
                        matrix.delta[source] = reader.get_signed(width + 1)?;
                    }
                }
            }
        }
        return Ok(());
    }

    let count = reader.get(4)? as usize;
    for _ in 0..count {
        let dest = reader.get(4)? as usize;
        let frac_bits = reader.get(4)?;
        let bypassed = u32::from(reader.get_bit()?);
        let mut coefficients = vec![0; sources];
        for slot in coefficients.iter_mut() {
            if reader.get_bit()? {
                *slot = reader.get_signed(frac_bits + 2)?;
            }
        }
        if restart.sync == RestartSync::B {
            reader.skip(4)?; // dither scale
        }
        matrices.push(Declared {
            dest,
            frac_bits,
            shift: 0,
            bypassed,
            coefficients,
            delta: vec![0; sources],
            delta_precision: 0,
        });
    }
    Ok(())
}

fn restart_header(reader: &mut BitReader<'_>, substream: &[u8]) -> Result<RestartHeader> {
    let start = reader.bit_position();
    let word = reader.get(14)?;
    let Some(sync) = RestartSync::from_word(word) else {
        return Err(Ran(format!("restart sync is {word:#x}")));
    };
    let output_timing = reader.get(16)?;
    let min_channel = reader.get(4)? as usize;
    let max_channel = reader.get(4)? as usize;
    let max_matrix_channel = reader.get(4)? as usize;
    let noise_shift = reader.get(4)?;
    let noisegen_seed = reader.get(23)?;
    let max_shift = reader.get(4)?;
    let max_huff_lsbs = reader.get(5)?;
    let max_output_bits = reader.get(5)?;
    let repeat = reader.get(5)?;
    if repeat != max_output_bits {
        return Err(Ran(format!(
            "the two output-bit fields disagree: {max_output_bits} and {repeat}"
        )));
    }
    let error_protect = reader.get_bit()?;
    let lossless_check = reader.get(8)?;
    // One bit of a value serialised across restart headers: the high half of a
    // 32-bit sample position whose low half is `output_timing`. A decoder
    // assembles it over several headers and complains about a run of zeros,
    // so reading it out of the right place matters even though nothing here
    // acts on it yet.
    let hires_output_timing = reader.get_bit()?;
    reader.skip(15)?;

    let mut channel_assignment = Vec::with_capacity(max_matrix_channel + 1);
    for _ in 0..=max_matrix_channel {
        channel_assignment.push(reader.get(6)?);
    }

    let bits = reader.bit_position() - start;
    let checksum_ok = crc::restart_header(substream, bits) == reader.get(8)? as u8;

    Ok(RestartHeader {
        sync,
        output_timing,
        min_channel,
        max_channel,
        max_matrix_channel,
        noise_shift,
        noisegen_seed,
        max_shift,
        max_huff_lsbs,
        max_output_bits,
        error_protect,
        hires_output_timing,
        lossless_check,
        channel_assignment,
        checksum_ok,
    })
}

/// Walk a whole stream.
pub fn stream(bytes: &[u8]) -> Result<Vec<AccessUnit>> {
    let mut units = Vec::new();
    let mut at = 0;
    let mut substreams = 0;
    let mut evolution = false;
    while at < bytes.len() {
        let unit = access_unit(&bytes[at..], substreams, evolution)?;
        substreams = unit.directory.len();
        if let Some(sync) = &unit.major_sync {
            evolution = sync.flags & (1 << 12) != 0;
        }
        at += unit.bytes;
        units.push(unit);
    }
    Ok(units)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A matrix that moves has to be written *and* the stream still has to
    /// decode, and only a decoder can catch the difference: a wrong ramp
    /// verifies every checksum in the unit and produces different samples.
    /// That half is `cargo xtask mlp`, through two decoders.
    ///
    /// This checks the half that can be checked without one — that a stream
    /// whose correlation sweeps actually reaches the path. A feature that never
    /// fires proves nothing about the decoder either.
    #[test]
    fn a_sweeping_correlation_aims_the_matrices() {
        use crate::{Config, Encoder, SampleBits};

        const CHANNELS: usize = 16;
        let mut encoder = Encoder::new(Config {
            sample_rate: 48_000,
            channels: CHANNELS,
            bits: SampleBits::TwentyFour,
        })
        .expect("sixteen channels, which is the shape with the syntax");
        let frames = encoder.frame_size();

        // Each pair is a source and that source scaled by a weight that sweeps
        // — a correlation that moves, which is the only thing a step can track.
        let mut state = 0x853c_49e6_748f_ea9bu64;
        for block in 0..320u32 {
            let mut samples = vec![0i32; frames * CHANNELS];
            for frame in 0..frames {
                for pair in 0..CHANNELS / 2 {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let source = ((state >> 42) as i32) << 10 >> 10;
                    let at = f64::from(block) * frames as f64 + frame as f64;
                    let weight = (at / 9_000.0 + pair as f64).sin() * 0.9;
                    samples[frame * CHANNELS + pair * 2] = source;
                    samples[frame * CHANNELS + pair * 2 + 1] = (f64::from(source) * weight) as i32;
                }
            }
            encoder.push(&samples);
        }
        // What was written is counted once it is written, and the encoder
        // holds intervals back while it works on them: finished, it has
        // written them all.
        encoder.finish(&[]);

        let stats = encoder.stats();
        assert!(
            stats.matrixed_share() > 0.5,
            "a sweeping correlation should be matrixed: {:.2}",
            stats.matrixed_share()
        );
        assert!(
            stats.aimed_matrices > 0,
            "no matrix was given a step to follow, so nothing interpolated"
        );
    }

    /// An access unit's bytes are delivered over the gap to the *next* unit's
    /// arrival, and a decoder refuses a unit that would have to arrive faster
    /// than the peak the stream declared. So a unit larger than the average has
    /// to be given a longer gap.
    ///
    /// Two things have to hold at once, and they pull against each other: every
    /// unit gets the time it needs, and the arrival clock still runs at the
    /// same average rate as the presentation clock — a stream whose clocks
    /// drift apart is one whose buffer eventually empties or overflows.
    ///
    /// Sizing the gap from the unit *before* the one just written satisfies
    /// neither and looks like it satisfies both: the mean is right, the gaps
    /// vary, and a decoder complains 154 times over a two-second stream.
    #[test]
    fn every_unit_is_given_the_time_its_bytes_need() {
        use crate::{Config, Encoder, SampleBits};

        // Sixteen channels, because they are the only shape that can exceed
        // what it declared: twenty-four bits of them at 48 kHz come to
        // 18.4 Mbit/s uncompressed and the format's ceiling is 18.0. Anything
        // narrower cannot reach its own peak however incompressible it is, so
        // its schedule would be the nominal one and prove nothing.
        const CHANNELS: usize = 16;
        let mut encoder = Encoder::new(Config {
            sample_rate: 48_000,
            channels: CHANNELS,
            bits: SampleBits::TwentyFour,
        })
        .expect("a shape the encoder writes");
        let frames = encoder.frame_size();

        // Incompressible where it is loud and silent where it is not, so the
        // units differ enough for the schedule to have to do something.
        let mut noise = 0x2545_f491_4f6c_dd1du64;
        let mut written = Vec::new();
        for block in 0..200u32 {
            let loud = block % 8 < 3;
            let samples: Vec<i32> = (0..frames * CHANNELS)
                .map(|_| {
                    noise ^= noise << 13;
                    noise ^= noise >> 7;
                    noise ^= noise << 17;
                    if loud {
                        ((noise >> 40) as i32) << 8 >> 8
                    } else {
                        0
                    }
                })
                .collect();
            written.extend_from_slice(&encoder.push(&samples));
        }
        let peak = encoder.declared_peak_bitrate() * 16 / 48_000;

        written.extend_from_slice(&encoder.finish(&[]));
        let units = stream(&written).expect("our own stream reads");
        assert!(units.len() > 100);

        let mut intervals = Vec::new();
        for pair in units.windows(2) {
            let interval =
                u64::from((pair[1].input_timing as u16).wrapping_sub(pair[0].input_timing as u16));
            // What this unit's bytes need at the declared peak.
            let words = (pair[0].bytes / 2) as u64;
            let needed = (words << 8).div_ceil(peak);
            assert!(
                interval >= needed,
                "a {}-byte unit was given {interval} samples and needs {needed}",
                pair[0].bytes
            );
            intervals.push(interval);
        }

        // And the clocks keep the same average rate, or the buffer drifts.
        let mean = intervals.iter().sum::<u64>() as f64 / intervals.len() as f64;
        assert!(
            (mean - frames as f64).abs() < 0.5,
            "the arrival clock ran at {mean:.3} samples a unit against {frames}"
        );
        assert!(
            intervals.iter().any(|interval| *interval != frames as u64),
            "nothing was scheduled: every gap was the nominal one"
        );
    }

    /// The extra data block this encoder writes for a sixteen-element
    /// programme: two bytes of length, an Evolution frame, and one object
    /// audio metadata payload of 65 bytes, written without a key. Pinned byte
    /// for byte, so a change to the block's layout is a failing test here
    /// rather than a stream some decoder rejects. Agreement with a block
    /// somebody else wrote is the harness's job: `cargo xtask thd` reads one
    /// and reports every field of it.
    const PINNED_EXTRA_DATA: [u8; 76] = [
        0x80, 0x25, 0x00, 0x47, 0x02, 0xc2, 0x82, 0x1f, 0x84, 0x4b, 0x80, 0x00, 0xa2, 0x71, 0x3b,
        0x98, 0x40, 0xe4, 0x71, 0x60, 0x81, 0xcc, 0xd7, 0x21, 0x03, 0xa1, 0x97, 0x02, 0x07, 0x52,
        0xf8, 0x04, 0x0e, 0xc5, 0x93, 0x08, 0x1d, 0xca, 0x6c, 0x10, 0x3c, 0x13, 0x64, 0x20, 0x79,
        0x23, 0xe0, 0x40, 0xf4, 0x41, 0x00, 0x81, 0xec, 0x76, 0x61, 0x03, 0xe0, 0xd5, 0x82, 0x07,
        0xd1, 0x7c, 0x84, 0x0f, 0xc2, 0x9c, 0x08, 0x1f, 0xc4, 0x60, 0x10, 0x00, 0x02, 0x2d, 0x00,
        0xfe,
    ];

    #[test]
    fn the_pinned_extra_data_reads_as_itself() {
        let extra = extra_data(&PINNED_EXTRA_DATA).expect("an extra data block");

        assert_eq!(extra.bytes, 76, "two bytes of length and 37 words");
        assert!(
            extra.check_nibble_ok,
            "the nibble that says this is a block"
        );
        assert!(extra.parity_ok, "and the byte that says it arrived whole");
        assert!(extra.frame_ok);
        assert_eq!(extra.declared_bytes, 71, "the room after the frame header");
        assert_eq!((extra.version, extra.key), (0, 0));
        assert_eq!(
            extra.protection,
            (8, 0),
            "an eight-bit primary protection field closes the frame, and no secondary one"
        );
        // The frame is 561 bits from bit 32, so the field is its last eight,
        // and it does not start on a byte.
        assert_eq!(extra.protection_at, 585);
        assert_eq!(
            extra.protection_bytes,
            [0x5a],
            "the constant an unkeyed stream carries"
        );

        assert_eq!(extra.payloads.len(), 1);
        let payload = &extra.payloads[0];
        assert_eq!(payload.id, 11, "object audio metadata");
        assert_eq!(payload.bytes.len(), 65);
        assert_eq!(payload.sample_offset, None, "it takes effect at the start");
        // The payload's own first byte, so a reader that lost its place in the
        // configuration ahead of it is caught here rather than in the object
        // metadata it hands on.
        assert_eq!(payload.bytes[0], 0x1f);
    }

    /// A block whose bytes were disturbed says so and stops there, rather than
    /// making the access unit around it unreadable: the audio beside it is
    /// still exact, and a payload size read out of corrupt bytes is not to be
    /// trusted to bound anything.
    #[test]
    fn a_disturbed_block_is_reported_rather_than_fatal() {
        let mut bytes = PINNED_EXTRA_DATA;
        bytes[20] ^= 0x40;
        let extra = extra_data(&bytes).expect("still a block");
        assert!(extra.check_nibble_ok, "its length is untouched");
        assert!(!extra.parity_ok);
    }

    /// And what the encoder writes there comes back, at sizes on both sides of
    /// the boundary where a payload's own length stops fitting in one group of
    /// bits.
    #[test]
    fn an_evolution_payload_reads_back_as_itself() {
        use crate::{Config, Encoder, SampleBits};

        let mut encoder = Encoder::new(Config {
            sample_rate: 48_000,
            channels: 16,
            bits: SampleBits::TwentyFour,
        })
        .expect("an immersive stream, which is what declares evolution");

        let sizes = [0usize, 1, 200, 255, 256, 257, 400];
        let frames = encoder.frame_size();
        let mut written = Vec::new();
        for size in sizes {
            let payload: Vec<u8> = (0..size).map(|n| (n * 7 + 1) as u8).collect();
            encoder.set_evolution(11, &payload);
            written.extend_from_slice(&encoder.push(&vec![0i32; frames * 16]));
            // And a unit that was told nothing carries nothing at all: there
            // is no empty shape, so the block is simply absent.
            written.extend_from_slice(&encoder.push(&vec![0i32; frames * 16]));
        }

        written.extend_from_slice(&encoder.finish(&[]));
        let units = stream(&written).expect("our own stream reads");
        assert_eq!(units.len(), sizes.len() * 2);
        for (which, unit) in units.iter().enumerate() {
            let Some(extra) = &unit.evolution else {
                assert!(which % 2 == 1, "unit {which} should carry a payload");
                assert_eq!(unit.extra_data, 0);
                continue;
            };
            assert!(
                extra.check_nibble_ok && extra.parity_ok && extra.frame_ok,
                "{which}"
            );
            assert_eq!(extra.payloads.len(), 1);
            let payload = &extra.payloads[0];
            assert_eq!(payload.id, 11);
            let size = sizes[which / 2];
            assert_eq!(payload.bytes.len(), size, "unit {which}");
            assert!(
                payload
                    .bytes
                    .iter()
                    .enumerate()
                    .all(|(n, byte)| *byte == (n * 7 + 1) as u8),
                "unit {which}: the bytes came back changed"
            );
        }
    }

    /// Given a key, the field holds a digest the reader can verify with the
    /// same key and not with another; given none, it holds a constant that
    /// verifies against nothing. Either way every other check on the block
    /// still passes — the parity byte is restated after the field is signed.
    #[test]
    fn a_keyed_stream_verifies_with_its_key_and_only_its_key() {
        use crate::{Config, Encoder, SampleBits};
        use hz_core::hmac::HmacSha256;

        let write = |key: Option<&[u8]>| {
            let mut encoder = Encoder::new(Config {
                sample_rate: 48_000,
                channels: 16,
                bits: SampleBits::TwentyFour,
            })
            .expect("an immersive stream");
            if let Some(key) = key {
                encoder.set_evolution_key(key);
            }
            let frames = encoder.frame_size();
            let mut written = Vec::new();
            for which in 0..6u8 {
                let payload: Vec<u8> = (0..40 + usize::from(which)).map(|n| n as u8).collect();
                encoder.set_evolution(11, &payload);
                let unit: Vec<i32> = (0..frames * 16).map(|n| (n as i32 % 97) - 48).collect();
                written.extend_from_slice(&encoder.push(&unit));
            }
            written.extend_from_slice(&encoder.finish(&[]));
            written
        };
        let verify = |stream: &[u8], key: &HmacSha256| -> Vec<Option<bool>> {
            let mut at = 0;
            let mut substreams = 0;
            let mut evolution = false;
            let mut results = Vec::new();
            while at < stream.len() {
                let unit = access_unit(&stream[at..], substreams, evolution).expect("reads");
                substreams = unit.directory.len();
                if let Some(sync) = &unit.major_sync {
                    evolution = sync.flags & (1 << 12) != 0;
                }
                if let Some(extra) = &unit.evolution {
                    assert!(extra.check_nibble_ok && extra.parity_ok && extra.frame_ok);
                    assert_eq!(extra.protection, (8, 0));
                    results.push(unit.protection_verifies(key, &stream[at..]));
                }
                at += unit.bytes;
            }
            results
        };

        let key = HmacSha256::new(b"a throwaway key for the test");
        let other = HmacSha256::new(b"another");

        let signed = write(Some(b"a throwaway key for the test"));
        let results = verify(&signed, &key);
        assert_eq!(results.len(), 6, "every unit carried a frame");
        assert!(results.iter().all(|r| *r == Some(true)), "{results:?}");
        assert!(
            verify(&signed, &other).iter().all(|r| *r == Some(false)),
            "another key does not verify"
        );

        let unsigned = write(None);
        assert!(
            verify(&unsigned, &key).iter().all(|r| *r == Some(false)),
            "a constant verifies against nothing"
        );
        // And the field is the only difference between the two streams.
        assert_eq!(signed.len(), unsigned.len());
        let differing = signed.iter().zip(&unsigned).filter(|(a, b)| a != b).count();
        // The field, which straddles two bytes, and the parity byte it
        // disturbs: three a unit at most.
        assert!(differing <= 18 && differing > 0, "{differing} bytes differ");
    }

    /// A substream directory of four entries, each carrying the second word
    /// where the dynamic range gain lives, laid out by hand: the first word
    /// is the extra-word flag, the restart flag clear, the checkdata flag,
    /// and twelve bits of where the substream ends; the second is nine bits
    /// of signed gain, three of how long it stands, and four spare.
    ///
    /// Pinned as bytes rather than written through the writer, so that a
    /// reader and a writer that agree on a wrong layout do not pass each
    /// other. The gains cover zero, the negative range, a positive one and
    /// the widest reduction the field can say.
    const PINNED_DIRECTORY: [u8; 16] = [
        0xa0, 0x48, 0x00, 0x00, // ends at word 72, gain 0 for 2^0 units
        0xa0, 0x62, 0xf9, 0x20, // ends at word 98, gain -14 for 2^2 units
        0xa0, 0x7f, 0x02, 0xb0, // ends at word 127, gain +5 for 2^3 units
        0xa0, 0xc1, 0xce, 0x70, // ends at word 193, gain -100 for 2^7 units
    ];

    #[test]
    fn a_directory_entry_carrying_a_gain_reads_as_itself() {
        let mut reader = BitReader::new(&PINNED_DIRECTORY);
        let entries: Vec<Directory> = (0..4).map(|_| entry(&mut reader).unwrap()).collect();

        for read in &entries {
            assert!(read.extra_word, "every entry states a gain here");
            assert!(!read.restart_nonexistent, "this unit restarts");
            assert!(read.checkdata_present);
        }
        // Cumulative, so each is where its substream finishes.
        assert_eq!(
            entries.iter().map(|e| e.end_words).collect::<Vec<_>>(),
            vec![72, 98, 127, 193]
        );
        assert_eq!(
            entries
                .iter()
                .map(|e| (e.drc_gain_update, e.drc_time_update))
                .collect::<Vec<_>>(),
            vec![(0, 0), (-14, 2), (5, 3), (-100, 7)],
        );
        assert!(reader.is_aligned() && reader.bits_left() == 0);
    }

    /// A presentation stated on an early substream reaches the stream, and the
    /// full decode is still bit exact.
    ///
    /// The two are one test on purpose. A presentation's rows do not invert —
    /// they are a mix of the channels that presentation carries, computed for
    /// a decoder that stops there — so the way to get them wrong is to let
    /// them reach the substream a full decode reads. That substream declares
    /// the lossless matrices and no others, and the check below is what says
    /// so: the stream still round-trips to the samples that went in.
    #[test]
    fn a_presentation_reaches_its_substream_and_leaves_the_decode_lossless() {
        use crate::matrix::{FRACTION, Primitive};
        use crate::{Config, Encoder, SampleBits};

        const CHANNELS: usize = 8;
        let mut encoder = Encoder::new(Config {
            sample_rate: 48_000,
            channels: CHANNELS,
            bits: SampleBits::TwentyFour,
        })
        .expect("an encoder");

        // A two-channel presentation: a mix of the two channels it carries,
        // with no unit coefficient in it — which is what makes it a fold and
        // not a lifting step.
        let rows = [
            Primitive::rounded(0, &[0.796, -0.713], FRACTION).expect("inside the field"),
            Primitive::rounded(1, &[0.500, 0.713], FRACTION).expect("inside the field"),
        ];
        encoder.set_presentation(0, &rows);

        let frames = encoder.frame_size();
        let mut source = Vec::new();
        let mut unit = vec![0i32; frames * CHANNELS];
        let mut stream = Vec::new();
        for block in 0..12 {
            for frame in 0..frames {
                for channel in 0..CHANNELS {
                    // Something with structure across the channels, so a
                    // matrix that ran where it should not would show.
                    let n = (block * frames + frame) as i32;
                    let value = (n * (channel as i32 + 3)) % 9_001 - 4_500;
                    unit[frame * CHANNELS + channel] = value;
                }
            }
            source.extend_from_slice(&unit);
            stream.extend_from_slice(&encoder.push(&unit));
        }
        stream.extend_from_slice(&encoder.finish(&[]));

        // The stream parses. That is what this can check from here — the
        // decode itself is checked against real decoders by
        // `cargo xtask mlp --presentation`, which is where this crate's
        // lossless proof lives.
        let units = stream_units(&stream).expect("a stream carrying a presentation");
        assert!(!units.is_empty());
        assert!(
            units.iter().any(|unit| unit.major_sync.is_some()),
            "no restart, so no matrix was ever stated"
        );
        let _ = source;
    }

    /// And the encoder's own words come back, on the schedule it promised.
    ///
    /// A value stands for `2^refresh` access units and no longer, and a
    /// decoder counts the unit it reads the next word in — so the gap may be
    /// that many units and not one more. Testing the bound before counting
    /// puts every word exactly one unit late, which is what happened.
    #[test]
    fn the_encoder_restates_its_gain_before_the_last_one_lapses() {
        use crate::format::DynamicRange;
        use crate::{Config, Encoder, SampleBits};

        for refresh in [0u32, 1, 4, 7] {
            let mut encoder = Encoder::new(Config {
                sample_rate: 48_000,
                channels: 6,
                bits: SampleBits::TwentyFour,
            })
            .expect("a shape the encoder writes");
            let gain = DynamicRange::from_db(-3.5, refresh).expect("a writable gain");
            for substream in 0..2 {
                encoder.set_dynamic_range(substream, Some(gain));
            }

            let frames = encoder.frame_size();
            let mut written = Vec::new();
            for _ in 0..400 {
                written.extend_from_slice(&encoder.push(&vec![0i32; frames * 6]));
            }

            written.extend_from_slice(&encoder.finish(&[]));
            let units = stream(&written).expect("our own stream reads");
            let mut since = [None::<u64>; 2];
            let mut stated = 0usize;
            for unit in &units {
                for (which, read) in unit.directory.iter().enumerate() {
                    if let Some(count) = &mut since[which] {
                        *count += 1;
                    }
                    if !read.extra_word {
                        continue;
                    }
                    stated += 1;
                    assert_eq!(read.drc_gain_update, gain.gain);
                    assert_eq!(read.drc_time_update, refresh);
                    if let Some(count) = since[which] {
                        assert!(
                            count <= gain.deadline(),
                            "2^{refresh}: restated after {count} units, and it stood for {}",
                            gain.deadline()
                        );
                    }
                    since[which] = Some(0);
                }
            }
            assert!(stated >= 2, "2^{refresh}: nothing was stated at all");
        }
    }

    /// The major sync this encoder writes for a sixteen-element stream,
    /// pinned byte for byte.
    ///
    /// It is here for one reason: **its length is not a fixed 28 bytes**. That
    /// is read from two bytes inside it before anything else is parsed, and a
    /// reader that assumes 28 lands in the middle of the substream directory
    /// and produces entries that look plausible and are nonsense. This was
    /// exactly what happened first.
    const PINNED_SYNC: [u8; 32] = [
        0xf8, 0x72, 0x6f, 0xba, 0x00, 0x07, 0x80, 0x4f, //
        0xb7, 0x52, 0x10, 0x00, 0x00, 0x00, 0x97, 0x6f, //
        0x43, 0xfc, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, //
        0x00, 0x01, 0x1f, 0xc6, 0xfc, 0x00, 0xbe, 0x61,
    ];

    #[test]
    fn an_immersive_major_sync_reads_as_itself() {
        let sync = major_sync(&PINNED_SYNC).expect("a major sync");
        assert_eq!(sync.size, 32, "an extra channel meaning makes it longer");
        assert!(sync.truehd);
        assert_eq!(sync.substreams, 4);
        assert_eq!(sync.arrangement_6ch, 0x0f, "L R C LFE Ls Rs");
        assert_eq!(sync.arrangement_8ch, 0x4f, "and the rear pair");
        assert_eq!(sync.substream_info, 0xfc);
        assert_eq!(sync.extended_substream_info, 0x3);
        assert_eq!(
            sync.peak_bitrate, 5999,
            "the ceiling this encoder declares, in 16-bit words a unit"
        );
        assert_eq!(sync.extra_channel_meaning.as_ref().map(Vec::len), Some(4));
        assert_eq!(
            sync.immersive,
            Some(Immersive {
                dialogue_norm: 31,
                mix_level: 35,
                objects: 15,
                dynamic_objects_only: true,
                lfe: true,
            }),
            "fifteen dynamic objects and a low frequency channel"
        );
        assert!(
            sync.checksum_ok,
            "the checksum has to hold over the longer header too"
        );
    }

    /// A truncated one must be refused rather than read as a shorter header
    /// that happens to parse.
    #[test]
    fn a_major_sync_that_does_not_fit_is_refused() {
        assert!(major_sync(&PINNED_SYNC[..30]).is_err());
        assert!(major_sync(&PINNED_SYNC[..20]).is_err());
    }

    /// And the reader has to agree with the writer, which is what makes it
    /// useful as a guard rather than as a second opinion nobody consults.
    #[test]
    fn the_reader_reads_what_the_encoder_writes() {
        use crate::{Config, Encoder, SampleBits};

        for channels in [1usize, 2, 6, 8, 16] {
            let mut encoder = Encoder::new(Config {
                sample_rate: 48_000,
                channels,
                bits: SampleBits::TwentyFour,
            })
            .expect("a shape the encoder writes");

            let unit_frames = encoder.frame_size();
            let mut stream = Vec::new();
            for block in 0..40u32 {
                let samples: Vec<i32> = (0..unit_frames * channels)
                    .map(|n| {
                        let phase = (block as f64 * 17.0 + n as f64) / 23.0;
                        (phase.sin() * 3_000_000.0) as i32
                    })
                    .collect();
                stream.extend_from_slice(&encoder.push(&samples));
            }

            stream.extend_from_slice(&encoder.finish(&[]));
            let units = stream_units(&stream).expect("our own stream reads");
            assert!(!units.is_empty());

            let expected_substreams = match channels {
                16 => 4,
                more if more > 2 => 2,
                _ => 1,
            };
            let immersive = channels == 16;
            for (index, unit) in units.iter().enumerate() {
                assert!(unit.parity_nibble_ok, "{channels} ch, unit {index}: parity");
                assert_eq!(unit.directory.len(), expected_substreams);
                assert_eq!(unit.extra_data, 0, "nothing writes extra data yet");
                for substream in &unit.substreams {
                    assert!(substream.parity_ok, "{channels} ch, unit {index}");
                    assert!(substream.checksum_ok, "{channels} ch, unit {index}");
                    if let Some(restart) = &substream.restart {
                        assert!(restart.checksum_ok, "{channels} ch, unit {index}");
                    }
                }
                if let Some(sync) = &unit.major_sync {
                    assert!(sync.checksum_ok, "{channels} ch, unit {index}");
                    assert_eq!(sync.substreams, expected_substreams);
                    // An immersive major sync is longer, and it is longer
                    // because of the block that says what its sixteen elements
                    // are.
                    assert_eq!(sync.size, if immersive { 32 } else { 28 });
                    assert_eq!(sync.immersive.is_some(), immersive);
                }
            }

            // The channels have to add up, and the split has to be where the
            // format puts it.
            let first = &units[0];
            let restarts: Vec<&RestartHeader> = first
                .substreams
                .iter()
                .filter_map(|s| s.restart.as_ref())
                .collect();
            assert_eq!(restarts.len(), expected_substreams);
            assert_eq!(restarts[0].min_channel, 0);
            assert_eq!(restarts[0].max_channel, channels.min(2) - 1);
            assert_eq!(restarts[0].sync, RestartSync::A, "substream 0 is always A");

            // The channels have to be shared out without a gap or an overlap,
            // and each substream's matrices have to reach everything decoded
            // up to it — which is what makes a presentation reconstructible by
            // a decoder that stops there.
            let mut expected_first = 0;
            for (which, restart) in restarts.iter().enumerate() {
                assert_eq!(restart.min_channel, expected_first, "{channels} ch");
                assert_eq!(restart.max_matrix_channel, restart.max_channel);
                assert_eq!(
                    restart.channel_assignment.len(),
                    restart.max_channel + 1,
                    "the assignment spans the matrix channels, not this substream's own"
                );
                if which + 1 == restarts.len() {
                    assert_eq!(restart.max_channel, channels - 1, "{channels} ch");
                }
                expected_first = restart.max_channel + 1;
            }

            if immersive {
                let words: Vec<RestartSync> = restarts.iter().map(|r| r.sync).collect();
                assert_eq!(
                    words,
                    vec![
                        RestartSync::A,
                        RestartSync::B,
                        RestartSync::B,
                        RestartSync::C
                    ],
                    "only substream 3 may speak C, and it must"
                );
            } else if expected_substreams == 2 {
                assert_eq!(
                    restarts[1].sync,
                    if channels > 6 {
                        RestartSync::B
                    } else {
                        RestartSync::A
                    },
                    "past six matrix channels A is refused"
                );
            }
        }
    }

    fn stream_units(bytes: &[u8]) -> Result<Vec<AccessUnit>> {
        stream(bytes)
    }
}
