// SPDX-License-Identifier: GPL-3.0-or-later
//
// Carries material ported from FFmpeg, which is licensed under the GNU Lesser
// General Public License, version 2.1 or later, and is used here under the
// GPL-3.0-or-later of this project as that licence permits:
//   libavcodec/mlpenc.c — Copyright (c) 2008 Ramiro Polla, 2016-2019 Jai Luthra
//   libavcodec/mlp.h — Copyright (c) 2007-2008 Ian Caulfield
// and from the `truehd` crate (https://github.com/truehdd/truehdd) —
// Copyright (c) the truehdd authors — under the Apache License, Version 2.0,
// whose text is reproduced in LICENSES/Apache-2.0.txt.
// What was taken and what was changed is recorded in docs/provenance.md.

//! Writing one access unit.
//!
//! An access unit is:
//!
//! ```text
//! ┌────────────────┬──────────────┬─────────────────┬──────────────┐
//! │ 4-byte header  │ major sync   │ substream       │ substream    │
//! │ length, parity │ (key units)  │ headers, 2 each │ payloads     │
//! └────────────────┴──────────────┴─────────────────┴──────────────┘
//! ```
//!
//! The header carries the total length and a parity nibble over itself and the
//! substream headers, and neither is known until the payload exists — so the
//! payload is written first and the header patched in afterwards.
//!
//! # Almost every field is optional
//!
//! A decoder keeps what it was last given, so a block that has nothing new to
//! say costs a bit rather than a description. What it may leave unsaid is
//! therefore a function of what the decoder is already holding, and that is
//! [`crate::model`]: the writer updates it as it emits, and the encoder asks
//! it whether a field has to be said at all. Being wrong about it does not
//! fail — it produces a stream that parses, whose every checksum verifies, and
//! which decodes to different samples — so the writer also asserts, in debug,
//! that every field it leaves unsaid is one the model says is already held.

use crate::crc;
use crate::filter::Coding;
use crate::format::{Coded, DynamicRange, RestartSync};
use crate::model::DecoderModel;
use hz_core::bits::{BitSink, BitWriter};

const SYNC_MAJOR: u32 = 0x00f8_726f;
const SYNC_TRUEHD: u32 = 0xba;
const MAJOR_SYNC_SIGNATURE: u32 = 0xb752;

/// The parameter-presence flags a decoder assumes after a restart header.
pub(crate) const PARAMS_DEFAULT: u8 = 0xff;
const PARAM_BLOCKSIZE: u8 = 1 << 7;
const PARAM_MATRIX: u8 = 1 << 6;
const PARAM_OUTSHIFT: u8 = 1 << 5;
const PARAM_QUANTSTEP: u8 = 1 << 4;
const PARAM_PRESENCE: u8 = 1 << 0;

/// The largest output shift a block can declare.
///
/// The field is four bits and signed, so seven, and a negative one would be a
/// decoder throwing away bits the encoder sent.
pub const MAX_OUTPUT_SHIFT: u8 = 7;

/// Bits per sample in the uncompressed coding. Also the `huff_lsbs` written
/// for every channel, and the width of the codec's internal sample.
pub(crate) const CODEC_BITS: u32 = 24;

/// One block within an access unit.
///
/// A unit is usually one block. A unit that carries a restart header is two,
/// and the reason is the whole point of a restart: it is where a decoder may
/// join the stream, so nothing after it may depend on what came before. The
/// filter state is what would — so the first block after a restart is
/// [`crate::encoder::RESTART_BLOCK`] samples long and coded with no prediction
/// at all, which refills that state from the block's own data, and the rest of
/// the unit follows with filters on.
///
/// The format says as much itself: after a restart header a decoder's block
/// size is eight and its filter orders are zero, and eight is exactly the
/// longest filter this format has.
pub(crate) struct Block<'a> {
    /// Where this block starts in the unit, and how long it is.
    pub first: usize,
    pub frames: usize,
    /// Whether the length has to be stated, which it does whenever it differs
    /// from the last block's. A restart header sets it back to eight, and a
    /// restarting unit's two blocks are eight and the rest — so the unit after
    /// one has to say so too, and saying nothing there left a decoder reading
    /// thirty-two samples out of a forty-sample block and everything after it
    /// as nonsense.
    pub write_frames: bool,
    /// How each channel is coded here: its filter, and how wide its residuals
    /// are.
    pub codings: &'a [Coding],
    /// Whether each channel's second filter has to be described here.
    pub write_fir: &'a [bool],
    /// Per channel, how far the decoder shifts the output left.
    pub output_shift: &'a [u8],
    /// Whether this block restates the shifts.
    pub write_shift: bool,
    pub write_iir: &'a [bool],
    /// Whether each channel says anything here at all. A channel whose
    /// filters, codebook and width are all what the decoder already holds
    /// costs one bit instead of eleven, and most channels in most blocks are
    /// exactly that.
    pub write_params: &'a [bool],
}

/// What changes from one access unit to the next.
pub(crate) struct Unit<'a> {
    /// Residuals, channel-major: `samples[channel][frame]`. Equal to the
    /// samples themselves for a channel whose filter is order zero.
    pub samples: &'a [Vec<i32>],
    /// The blocks this unit is written as, in order.
    pub blocks: &'a [Block<'a>],
    /// The primitive matrices a decoder applies after filtering, in order.
    pub matrices: &'a [crate::matrix::Primitive],
    /// The steps that turn the stored channels back into the elements, which
    /// only the last substream declares — and after the coding matrices, since
    /// a decoder applies them in the order they are written and the coding ones
    /// have to come off first. An early-stopping decoder must never see these:
    /// the stored channels *are* its presentation, and undoing the arrangement
    /// would take it apart.
    pub arrangement: &'a [crate::matrix::Primitive],
    /// Which output channel each of a substream's matrix channels is, where it
    /// is not simply itself.
    ///
    /// A presentation's matrix **permutes**: the 5.1's left is built where the
    /// stored row it is made of sits, not where the 5.1 wants it. Saying that
    /// permutation in matrices costs three of them per transposition — 23 rows
    /// for a 5.1 against the 15 a substream may declare — and the format has a
    /// field for it. Empty for a substream that wants the identity, which the
    /// last one always does: a full decode hands back the elements in order.
    pub assignment: &'a [Vec<u8>],
    /// What each early substream states as its own output shift.
    ///
    /// 🔴 Not what the encoder shifted by. A presentation's shift only says how
    /// loud its own output comes out, so it is free — and the reference
    /// restates it in every substream for exactly that reason, its stereo
    /// saying `[3, 4]` for the same two channels its 5.1 says `[3, 2]` for.
    /// What the encoder really shifted by is what the *last* substream states,
    /// and that is what makes a full decode lossless.
    pub presentation_shift: &'a [Vec<u8>],
    /// What each substream before the last declares instead, when it declares
    /// something of its own: the rows that **compute a presentation** from the
    /// internal channels, rather than a subset of the coding matrices. Empty
    /// for a substream that has none, which is what an encoder writing only
    /// lossless steps leaves everywhere. See [`declared_by`].
    pub presentations: &'a [Vec<crate::matrix::Primitive>],
    /// Whether they have to be said at all: a unit with none, following one
    /// with none, can stay quiet.
    pub write_matrix: bool,
    /// The decoding timestamp written in the access unit header.
    pub input_timing: u16,
    /// The presentation timestamp written in the restart header.
    pub output_timing: u16,
    /// The lossless check of everything since the previous restart header,
    /// per substream, which is what each one certifies.
    pub lossless_check: [u8; crate::format::MAX_SUBSTREAMS],
    /// This restart header's bit of the high-resolution timing field.
    pub hires_timing: bool,
    /// The largest dead-bit shift and the widest output sample anywhere in the
    /// interval this header opens. Only read when the unit restarts.
    pub max_shift: u8,
    pub max_output_bits: u8,
    /// How many of this unit's samples are padding, for a final short unit.
    pub shorten_by: u16,
    /// Whether this unit restarts the decoder: a major sync and a restart
    /// header, and everything after it decodable without what came before.
    pub restart: bool,
    /// The dynamic range each substream's directory entry states here, where
    /// it states one. An entry that carries a gain is four bytes rather than
    /// two, which is the only thing in this header that is not a fixed width.
    pub dynamic_range: [Option<DynamicRange>; crate::format::MAX_SUBSTREAMS],
    /// The gain a decoder joining at this unit's major sync applies until an
    /// update reaches it. Only read when the unit restarts.
    pub start_up_gain: i32,
    /// What to carry after the last substream, as an Evolution payload.
    pub evolution: Option<Evolution<'a>>,
    /// The key that signs the Evolution frame's protection field, when the
    /// encoder was given one. Without it the field holds a constant. See
    /// [`crate::protection`].
    pub protection: Option<&'a hz_core::hmac::HmacSha256>,
}

/// Write one access unit.
///
/// A unit that restarts the decoder carries a major sync and a restart header
/// — thirty bytes that make it independently decodable, and thirty bytes that
/// are pure overhead in every unit that does not need to be. At forty samples
/// a unit that is a large fraction of the stream: measured on a programme's
/// centre channel with prediction on, framing was 45 % of the bytes. So a
/// restart happens on an interval, and the units between one and the next
/// carry only their own parameters and their residuals.
pub(crate) fn access_unit(coded: &Coded, unit: &Unit<'_>, model: &mut DecoderModel) -> Vec<u8> {
    let mut writer = BitWriter::with_capacity(8192);

    // The header and the substream headers are patched in at the end.
    let header_at = writer.placeholder(4);
    if unit.restart {
        write_major_sync(&mut writer, coded, unit.start_up_gain);
    }
    let directory_at = writer.placeholder(directory_bytes(coded, unit));

    let payload_start = writer.byte_position();
    let mut ends = [0usize; crate::format::MAX_SUBSTREAMS];
    for (index, end) in ends.iter_mut().enumerate().take(coded.substreams) {
        let bytes = substream(coded, unit, index, model);
        writer.put_bytes(&bytes);
        // Each substream's `end` is where *it* finishes, counted in words from
        // the start of the payload — so they accumulate rather than each
        // measuring its own length.
        *end = (writer.byte_position() - payload_start) / 2;
    }

    let extra = unit
        .evolution
        .map(|evolution| write_extra_data(&mut writer, &evolution));

    let total = writer.byte_position();
    debug_assert!(
        total.is_multiple_of(2),
        "an access unit is measured in 16-bit words"
    );

    // Bit 15 says a second word follows carrying dynamic range control. Bit 14
    // says the substream does *not* restart the decoder, and a decoder rejects
    // a unit whose major sync and this bit disagree. Bit 13 says the parity
    // and checksum are present.
    const EXTRA_WORD: u16 = 1 << 15;
    const CHECKDATA_PRESENT: u16 = 1 << 13;
    let nonrestart = if unit.restart { 0 } else { 1 << 14 };
    let directory_end = directory_at + directory_bytes(coded, unit);
    let bytes = writer.bytes_mut();
    let mut at = directory_at;
    for (index, end) in ends.iter().enumerate().take(coded.substreams) {
        let drc = unit.dynamic_range[index];
        let extra = if drc.is_some() { EXTRA_WORD } else { 0 };
        let header: u16 = extra | nonrestart | CHECKDATA_PRESENT | (*end as u16 & 0x0fff);
        bytes[at..at + 2].copy_from_slice(&header.to_be_bytes());
        at += 2;
        if let Some(drc) = drc {
            // Nine bits of signed gain, three of how long it stands, and four
            // that are not used.
            let word = ((drc.gain as u16 & 0x01ff) << 7) | ((drc.refresh as u16 & 7) << 4);
            bytes[at..at + 2].copy_from_slice(&word.to_be_bytes());
            at += 2;
            model.state_range(index, drc);
        }
    }
    debug_assert_eq!(at, directory_end);

    // The access unit header. Its parity nibble covers the timing, the length
    // and the whole directory — everything in the unit that a decoder needs
    // before it can find anything else.
    let length = (total / 2) as u16;
    let mut parity = unit.input_timing ^ length;
    for pair in bytes[directory_at..directory_end].chunks(2) {
        parity ^= u16::from(pair[0]);
        parity ^= u16::from(pair[1]);
    }
    parity ^= parity >> 8;
    parity ^= parity >> 4;
    let header = ((parity & 0xf) ^ 0xf) << 12 | (length & 0x0fff);
    bytes[header_at..header_at + 2].copy_from_slice(&header.to_be_bytes());
    bytes[header_at + 2..header_at + 4].copy_from_slice(&unit.input_timing.to_be_bytes());

    // The protection field, last of all: its digest covers the header and the
    // directory just patched in.
    if let (Some(key), Some(extra)) = (unit.protection, extra) {
        sign_extra_data(bytes, key, extra);
    }

    writer.finish()
}

/// Where a unit's extra data block was written, for the digest that has to
/// wait until the unit's header is final.
#[derive(Clone, Copy)]
struct ExtraDataAt {
    /// The block's first byte.
    start: usize,
    /// The byte length its Evolution header declares.
    declared: usize,
    /// The bit, counted from the unit's start, at which the primary
    /// protection field's own bits begin.
    protection_bit: usize,
}

/// Fill the protection field with the leading byte of the digest, and restate
/// the block's parity byte, which the field is inside of.
fn sign_extra_data(bytes: &mut [u8], key: &hz_core::hmac::HmacSha256, at: ExtraDataAt) {
    let digest = crate::protection::digest(key, bytes, at.start, at.declared, at.protection_bit);
    overwrite_byte(bytes, at.protection_bit, digest[0]);

    let parity_at = at.start + crate::protection::FRAME_HEADER_BYTES + at.declared;
    bytes[parity_at] = bytes[at.start + 2..parity_at]
        .iter()
        .fold(EXTRA_DATA_PARITY_XOR, |acc, byte| acc ^ byte);
}

/// Overwrite eight bits at `bit`, which need not start on a byte.
fn overwrite_byte(bytes: &mut [u8], bit: usize, value: u8) {
    let at = bit / 8;
    let shift = (bit % 8) as u32;
    if shift == 0 {
        bytes[at] = value;
        return;
    }
    bytes[at] = (bytes[at] & (0xff << (8 - shift))) | (value >> shift);
    bytes[at + 1] = (bytes[at + 1] & (0xff >> shift)) | (value << (8 - shift));
}

/// Bytes the substream directory occupies, which is not fixed: an entry that
/// states a dynamic range gain is four bytes rather than two.
fn directory_bytes(coded: &Coded, unit: &Unit<'_>) -> usize {
    unit.dynamic_range[..coded.substreams]
        .iter()
        .map(|drc| if drc.is_some() { 4 } else { 2 })
        .sum()
}

fn write_major_sync(writer: &mut BitWriter, coded: &Coded, start_up_gain: i32) {
    let start = writer.byte_position();

    writer.put(24, SYNC_MAJOR);
    writer.put(8, SYNC_TRUEHD);

    writer.put(4, coded.rate);
    writer.put(1, 0); // 6-channel multichannel type
    writer.put(1, 0); // 8-channel multichannel type
    writer.put(2, 0); // ignored
    writer.put(2, coded.presentation_mod);
    writer.put(2, coded.presentation_mod);
    writer.put(5, coded.arrangement_6ch);
    writer.put(2, coded.presentation_mod);
    writer.put(13, coded.arrangement_8ch);

    writer.put(16, MAJOR_SYNC_SIGNATURE);
    writer.put(16, coded.flags);
    writer.put(16, 0); // ignored
    writer.put(1, 1); // variable bitrate
    writer.put(15, coded.peak_bitrate);
    writer.put(4, coded.substreams as u32);
    writer.put(2, 0); // ignored
    writer.put(2, coded.extended_substream_info);

    // channel_meaning: sixty-four bits describing presentations this encoder
    // does not author. Written as zero, which is what "not indicated" is.
    writer.put(8, u32::from(coded.substream_info));
    writer.put(6, 0); // reserved
    writer.put(1, 0); // 2-channel control enabled
    writer.put(1, 0); // 6-channel control enabled
    writer.put(1, 0); // 8-channel control enabled
    writer.put(1, 0); // reserved
    writer.put_signed(7, start_up_gain);
    writer.put(6, 31); // 2-channel dialogue normalisation: 31 is "none", 0 is a 31 dB cut
    writer.put(6, 0); // 2-channel mix level
    writer.put(5, 31); // 6-channel dialogue normalisation
    writer.put(6, 0); // 6-channel mix level
    writer.put(5, 0); // 6-channel source format
    writer.put(5, 31); // 8-channel dialogue normalisation
    writer.put(6, 0); // 8-channel mix level
    writer.put(6, 0); // 8-channel source format
    writer.put(1, 0); // reserved
    writer.put(1, u32::from(coded.immersive.is_some()));

    debug_assert!(writer.is_aligned() && writer.byte_position() - start == 26);
    if let Some(immersive) = &coded.immersive {
        write_extra_channel_meaning(writer, immersive);
    }
    let checksum = crc::major_sync(&writer.bytes()[start..]);
    writer.put(8, u32::from(checksum[0]));
    writer.put(8, u32::from(checksum[1]));
}

/// What the sixteen-element presentation is.
///
/// The major sync's length is not a field of its own: a reader works it out
/// from the presence bit above and from this block's own length field, before
/// it has parsed anything else. So an immersive major sync is longer than a
/// channel-based one, and this is what makes it longer.
///
/// The block is padded to a whole number of 16-bit words, which is what its
/// length field counts — one less than the number of words, measured from the
/// length field itself.
fn write_extra_channel_meaning(writer: &mut BitWriter, immersive: &crate::format::Immersive) {
    const WORDS: u32 = 2;
    const BITS: usize = WORDS as usize * 16;

    let start = writer.bit_position();
    writer.put(4, WORDS - 1);
    writer.put(5, immersive.dialogue_norm);
    writer.put(6, immersive.mix_level);
    writer.put(5, immersive.objects);
    // Every element is a dynamic object, so there is no fixed bed to describe
    // — only whether one of them is the low frequency channel.
    writer.put(1, 1); // dynamic objects only
    writer.put(1, u32::from(immersive.lfe));

    let written = writer.bit_position() - start;
    debug_assert!(
        written <= BITS,
        "the extra channel meaning overran its words"
    );
    writer.put((BITS - written) as u32, 0);
}

/// What rides after the last substream, when anything does.
///
/// A decoder enters this whenever the access unit has a whole 16-bit word left
/// after the substreams, so writing nothing is how a unit says it carries
/// nothing — there is no "empty" shape to write. What it finds there depends
/// on the major sync's flags rather than on anything in the block itself: with
/// bit 12 set it is an Evolution frame, and this writes only that shape.
///
/// ```text
/// ┌────────┬──────────┬────────────┬──────────┬───────────┬─────────┬────────┐
/// │ check  │ length   │ reserved   │ frame    │ Evolution │ zeroes  │ parity │
/// │ nibble │ 12 bits  │ 4 bits     │ 12 bits  │ frame     │         │ 8 bits │
/// └────────┴──────────┴────────────┴──────────┴───────────┴─────────┴────────┘
/// └── not counted by ──┘└──────── `length` 16-bit words ────────────────────┘
/// ```
///
/// The length does not count the two bytes that state it, and the frame's
/// declared byte length is the room after the Evolution header rather than the
/// frame's own size — a reader parses the frame structurally and pads to the
/// end, so the field is a bound. Both are what the reference does, and writing
/// them its way reproduces its block byte for byte.
///
/// Returns where the block went, for the digest that is written into it once
/// the unit's header is final.
fn write_extra_data(writer: &mut BitWriter, evolution: &Evolution<'_>) -> ExtraDataAt {
    debug_assert!(writer.is_aligned());
    let start = writer.byte_position();

    let frame_bits = evolution_bits(evolution);
    // The evolution header and the parity byte, then whatever it takes to
    // reach a whole word.
    let need = 24 + frame_bits;
    let total_bits = need.next_multiple_of(16);
    let words = total_bits / 16;
    let declared = (total_bits - 24) / 8;

    writer.put(4, check_nibble(words as u32));
    writer.put(12, words as u32);

    writer.put(4, 0); // reserved
    writer.put(12, declared as u32);

    let protection_bit = write_evolution(writer, evolution);
    for _ in 0..total_bits - 24 - frame_bits {
        writer.put(1, 0);
    }

    // Every byte of the block but the two that state its length and the parity
    // byte itself, folded together with a constant.
    let body = &writer.bytes()[start + 2..];
    let parity = body
        .iter()
        .fold(EXTRA_DATA_PARITY_XOR, |acc, byte| acc ^ byte);
    writer.put(8, u32::from(parity));

    debug_assert_eq!(writer.byte_position() - start, 2 + total_bits / 8);
    ExtraDataAt {
        start,
        declared,
        protection_bit,
    }
}

/// The seed the block's parity byte folds from.
const EXTRA_DATA_PARITY_XOR: u8 = 0xa9;

/// The nibble that makes the block's first two bytes fold to `0xf`.
fn check_nibble(words: u32) -> u32 {
    let length = ((words >> 8) as u8) ^ (words as u8);
    u32::from(0xf ^ (length ^ (length >> 4)) & 0xf)
}

/// An Evolution payload: what it is, its bytes, and when in the unit it takes
/// effect. The frame it goes in is the one E-AC-3 carries too, so it is
/// written by [`hz_meta::emdf`]; what is TrueHD's is the block around it.
pub type Evolution<'a> = hz_meta::emdf::Payload<'a>;

/// One Evolution frame carrying one payload, closed the way every reference
/// stream closes it: an eight-bit primary protection field and no secondary
/// one.
///
/// The field holds a keyed digest whose computation the format leaves to the
/// implementation, which no public decoder checks and which a decoder is told
/// it may ignore. It cannot be written here: the digest covers the unit's
/// header, which is patched in last, so the frame carries a constant and
/// [`sign_extra_data`] overwrites it once the unit is final — when there is a
/// key.
///
/// Returns the bit at which the field's own bits begin.
fn write_evolution(writer: &mut BitWriter, evolution: &Evolution<'_>) -> usize {
    hz_meta::emdf::write(
        writer,
        std::slice::from_ref(evolution),
        hz_meta::emdf::Protection::TRUEHD,
    )
    .primary
}

/// Bits one Evolution frame carrying `evolution` occupies.
fn evolution_bits(evolution: &Evolution<'_>) -> usize {
    hz_meta::emdf::bits(
        std::slice::from_ref(evolution),
        hz_meta::emdf::Protection::TRUEHD,
    )
}

/// One substream, written into its own buffer.
///
/// Its own buffer and not the access unit's, because the restart header's
/// checksum is defined over a bit range measured from the substream's first
/// byte, and its parity and checksum cover the substream and nothing else.
fn substream(coded: &Coded, unit: &Unit<'_>, index: usize, model: &mut DecoderModel) -> Vec<u8> {
    let mut writer = BitWriter::with_capacity(4096);
    let range = coded.ranges[index];

    for (which, block) in unit.blocks.iter().enumerate() {
        writer.put(1, 1); // a block header follows
        if which == 0 && unit.restart {
            writer.put(1, 1); // and it starts with a restart header
            write_restart_header(&mut writer, unit, index, &range);
            // A restart header is where a decoder may join, so it throws away
            // everything the blocks before it left standing.
            model.restart(index, &range);
        } else {
            writer.put(1, 0);
        }
        write_decoding_params(
            &mut writer,
            unit,
            block,
            coded,
            index,
            &range,
            which == 0,
            model,
        );
        write_block_data(&mut writer, unit, block, &range);
        writer.put(1, u32::from(which + 1 == unit.blocks.len()));
    }

    // The parity and checksum are the last two bytes, and the substream is
    // measured in 16-bit words, so the padding has to leave room for them.
    writer.align_to(16);
    let _ = coded;

    // A final short unit says so here: the marker is only looked for when
    // there are more than two bytes left after the alignment.
    if unit.shorten_by > 0 {
        for field in crate::encoder::end_of_stream(unit.shorten_by) {
            writer.put(16, field);
        }
    }

    let parity = crc::parity(writer.bytes()) ^ crc::PARITY_XOR;
    let checksum = crc::substream(writer.bytes());
    writer.put(8, u32::from(parity));
    writer.put(8, u32::from(checksum));

    writer.finish()
}

fn write_restart_header(
    writer: &mut BitWriter,
    unit: &Unit<'_>,
    index: usize,
    range: &crate::format::Substream,
) {
    let start = writer.bit_position();
    debug_assert_eq!(
        start, 2,
        "the restart header's checksum is defined from two bits into the \
         substream, so it has to begin there"
    );

    writer.put(14, range.sync.word());
    writer.put(16, u32::from(unit.output_timing));
    writer.put(4, range.first as u32);
    writer.put(4, range.last as u32);
    writer.put(4, range.max_matrix as u32);
    writer.put(4, 0); // noise shift
    writer.put(23, 0); // noise generator seed
    // The bound on the shifts the blocks after this may declare, and nothing
    // else: a decoder checks each against it and stores it. The encoder holds
    // the whole interval before writing any of it, and the shifts are settled
    // while it is prepared, so this is the largest the interval actually uses
    // rather than the largest the field can say.
    writer.put(4, u32::from(unit.max_shift));

    // `max_huff_lsbs` is the format's own maximum and not the interval's. It
    // is a bound on the residual widths, which are not known until the codings
    // are chosen, and those are chosen unit by unit *after* this header is
    // written. Declaring it tightly would mean choosing every coding in the
    // interval first, against a second copy of the decoder model — the field
    // is a fixed five bits either way, so it would buy no bytes at all, and a
    // duplicated model is what `crate::model` exists to have got rid of.
    writer.put(5, CODEC_BITS); // max_huff_lsbs
    // `max_output_bits` is the interval's: the samples a decoder emits are the
    // samples that went in, and the encoder is holding all of them. Except
    // where this substream puts out a presentation, which is a fold of them
    // and may be wider than any one; the field is fixed width, so it is told
    // the whole domain rather than measured, and a decoder that counts the
    // bits its outputs use — the reference one does — has nothing to say.
    let stated = unit
        .presentations
        .get(index)
        .is_some_and(|rows| !rows.is_empty());
    let output_bits = if stated {
        CODEC_BITS
    } else {
        u32::from(unit.max_output_bits)
    };
    writer.put(5, output_bits);
    writer.put(5, output_bits); // and again, as the format asks
    writer.put(1, 0); // no data check
    writer.put(8, u32::from(unit.lossless_check[index]));
    // One bit of the high-resolution output timing field, which is serialised
    // across restart headers; see [`crate::hires`]. Every substream carries
    // the same bit, because every substream's decoder assembles the same
    // value from its own headers.
    writer.put(1, u32::from(unit.hires_timing));
    writer.put(15, 0); // reserved, and no heavy dynamic range

    // The channel assignment, over the matrix span rather than this
    // substream's own channels. Identity unless this substream states one.
    let stated = unit.assignment.get(index);
    for channel in 0..=range.max_matrix {
        let output = stated
            .and_then(|order| order.get(channel))
            .map_or(channel, |output| usize::from(*output));
        writer.put(6, output as u32);
    }

    let bits = writer.bit_position() - start;
    // The checksum covers the restart header from two bits into the
    // substream's first byte, so it reads the substream buffer itself — and it
    // reaches into the byte still being assembled, which is why it needs the
    // padded view rather than the whole bytes.
    let checksum = crc::restart_header(&writer.padded(), bits);
    writer.put(8, u32::from(checksum));
}

/// The parameters for this block.
///
/// After a restart header a decoder assumes exactly what this encoder wants —
/// no matrices, no shift, no quantisation, filters off, codebook 0 and
/// twenty-four LSBs — so the only thing that has to be said is how many
/// samples the block holds. Everything else is signalled as unchanged, which
/// costs one bit each.
#[allow(clippy::too_many_arguments)]
fn write_decoding_params<S: BitSink>(
    writer: &mut S,
    unit: &Unit<'_>,
    block: &Block<'_>,
    coded: &Coded,
    index: usize,
    range: &crate::format::Substream,
    first: bool,
    model: &mut DecoderModel,
) {
    let flags = PARAMS_DEFAULT;

    if flags & PARAM_PRESENCE != 0 {
        writer.put(1, 0); // keep the default flags
    }
    if flags & PARAM_BLOCKSIZE != 0 {
        if block.write_frames {
            writer.put(1, 1);
            writer.put(9, block.frames as u32);
            model.state_blocksize(index, block.frames);
        } else {
            writer.put(1, 0);
            debug_assert_eq!(
                model.blocksize(index),
                block.frames,
                "a block said nothing about a length the decoder does not hold"
            );
        }
    }
    if flags & PARAM_MATRIX != 0 {
        // The matrices belong to the unit rather than to a block, and a
        // decoder holds them until they are replaced — so the first block says
        // them and the second says nothing.
        if unit.write_matrix && first {
            writer.put(1, 1);
            write_matrix_params(writer, &declared_by(unit, coded, index, range), range);
        } else {
            writer.put(1, 0);
        }
    }
    if flags & PARAM_OUTSHIFT != 0 {
        // The content's dead low bits, taken off before coding and put back
        // by the decoder after everything else — after the filters and after
        // the matrices, so it costs the prediction nothing. A decoder reads
        // one value per channel this substream's restart header covers, and
        // holds them until they are said again.
        let stated = unit
            .presentation_shift
            .get(index)
            .filter(|shifts| !shifts.is_empty());
        if block.write_shift || stated.is_some() {
            writer.put(1, 1);
            let shifts: Vec<u8> = (0..=range.max_matrix)
                .map(|channel| match stated {
                    Some(own) => own.get(channel).copied().unwrap_or(0),
                    None => block.output_shift[channel],
                })
                .collect();
            for shift in &shifts {
                writer.put_signed(4, i32::from(*shift));
            }
            // 🔴 The model holds what the *last* substream's decoder holds.
            // A real decoder keeps its shifts per substream, and an early
            // substream stating a presentation's own says them in every
            // block — so what it says must not be written into the model,
            // where the last substream, which said nothing about shifts
            // because the decoder already held them, would then be told it
            // had. A false alarm in debug builds, and a stream that decodes
            // in every build.
            if stated.is_none() {
                model.state_shifts(&shifts, range.max_matrix);
            }
        } else {
            writer.put(1, 0);
            debug_assert!(
                !model.restates_shifts(block.output_shift, range.max_matrix),
                "a block said nothing about shifts the decoder does not hold"
            );
        }
    }
    if flags & PARAM_QUANTSTEP != 0 {
        writer.put(1, 0);
    }

    // A decoder reads one bit per channel unconditionally, whatever the
    // presence flags say — for the channels *this substream codes*.
    for channel in range.first..=range.last {
        let coding = &block.codings[channel];
        // A decoder keeps everything it holds for a channel that says
        // nothing — both filters, the offset, the codebook and the width —
        // and a restart header clears all of it, so "nothing" is only ever
        // right between restarts, exactly as "unchanged" is for the filters
        // below.
        if !block.write_params[channel] {
            writer.put(1, 0);
            debug_assert!(
                !model.restates(channel, coding),
                "channel {channel} said nothing about a coding the decoder does not hold"
            );
            continue;
        }
        writer.put(1, 1); // this channel's coding follows

        // Both prediction filters, each with a bit saying whether this block
        // restates it. A decoder keeps what it was given until told otherwise
        // and a restart header clears both, so "unchanged" is only ever right
        // between restarts — being wrong here means a decoder predicting with
        // taps that are no longer there, and every checksum still verifying.
        if block.write_fir[channel] {
            writer.put(1, 1);
            write_filter_params(writer, &coding.filter.fir, coding.filter.shift);
        } else {
            writer.put(1, 0);
        }
        if block.write_iir[channel] {
            writer.put(1, 1);
            write_filter_params(writer, &coding.filter.iir, coding.filter.shift);
        } else {
            writer.put(1, 0);
        }

        // No offset on the residuals. The format carries a fifteen-bit one
        // that slides the codebook's window; it was implemented, measured, and
        // found to produce a larger stream than leaving it alone — see
        // `filter::best_entropy`.
        writer.put(1, 0);

        writer.put(2, u32::from(coding.codebook));
        writer.put(5, coding.huff_lsbs);
        model.state(channel, coding);
    }
}

/// Which matrices a substream declares.
///
/// A full decode applies the matrices of the **last** substream and of no
/// other, so that one declares every matrix in the unit — including those
/// writing channels an earlier substream coded. A substream before it declares
/// only what a decoder stopping there would need, which is what makes the
/// two-channel presentation decode correctly on its own.
///
/// That freedom is also what a *presentation* is. An earlier substream need
/// not declare a subset of the coding matrices at all: it can declare rows
/// that compute a fold of every internal channel into its own, and a decoder
/// stopping there hands back that fold. Those rows do not invert and do not
/// have to, since no full decode ever reads them. Without them the narrow
/// presentations are the leading channels as they were written, which for an
/// object programme means a stereo decoder hears two objects.
///
/// Getting this wrong is invisible in the bitstream: it parses, every checksum
/// verifies, and whole restart intervals of the stereo pair come back still
/// matrixed because nothing undid them.
fn declared_by<'a>(
    unit: &'a Unit<'_>,
    coded: &Coded,
    index: usize,
    range: &crate::format::Substream,
) -> Vec<&'a crate::matrix::Primitive> {
    let last = index + 1 == coded.substreams;
    // A presentation, where one is stated: the rows that compute it, and not
    // the coding matrices at all. They are not lifting steps and nothing
    // inverts them — a full decode reads the last substream's matrices, which
    // are still the lossless ones.
    // 🔴 The coding matrices come first, and leaving them out is silent. A
    // decoder stopping here applies *only* what this substream declares, and
    // the channels it has are still in the domain the coding matrices put them
    // in — so a presentation stated on its own is stated over the wrong
    // signal. Measured: a 5.1 whose levels were within a decibel of the
    // master's and whose channels correlated at 0.2.
    if !last
        && let Some(rows) = unit.presentations.get(index)
        && !rows.is_empty()
    {
        return unit
            .matrices
            .iter()
            .filter(|matrix| matrix.dest <= range.last)
            .chain(rows.iter())
            .collect();
    }
    unit.matrices
        .iter()
        .filter(|matrix| last || matrix.dest <= range.last)
        .chain(unit.arrangement.iter().filter(|_| last))
        .collect()
}

/// The primitive matrices, and their coefficients.
///
/// Coefficients are written for two more channels than the stream carries.
/// Those are the noise channels a decoder synthesises when the restart
/// header's noise type says so — which is the type this encoder declares, and
/// which FFmpeg's writes too. Their coefficients are zero; their presence bits
/// are not optional.
pub(crate) fn write_matrix_params<S: BitSink>(
    writer: &mut S,
    declared: &[&crate::matrix::Primitive],
    range: &crate::format::Substream,
) {
    if range.sync == RestartSync::C {
        write_immersive_matrix_params(writer, declared, range);
        return;
    }

    writer.put(4, declared.len() as u32);
    let sources = range.matrix_sources();

    for matrix in declared {
        debug_assert_eq!(
            matrix.shift, 0,
            "only restart sync word C can scale a matrix, and this one is declared elsewhere"
        );
        writer.put(4, matrix.dest as u32);
        writer.put(4, matrix.frac_bits);
        writer.put(1, 0); // no bypassed least significant bit

        for source in 0..sources {
            let coefficient = matrix.coefficients[source];
            if coefficient == 0 {
                writer.put(1, 0);
            } else {
                writer.put(1, 1);
                writer.put_signed(matrix.frac_bits + 2, matrix.stored(source));
            }
        }
        if range.matrix_dither() {
            writer.put(4, 0); // this matrix adds no dither
        }
    }
}

/// The same matrices, in the syntax restart sync word C speaks.
///
/// Three things are said differently and one thing is said that the other
/// syntaxes cannot say at all:
///
/// - Which coefficients are present is a **mask** written up front, one bit
///   per source, rather than a presence bit before each one. Same bits, other
///   order.
/// - The count is written as one less than itself, so this syntax cannot
///   express "no matrices" that way. It says so with the leading bit instead.
/// - Each matrix carries a shift applied to the whole of it, `2^n` on every
///   coefficient at once, on top of its fractional bits.
/// - And a second set of coefficients may be given that the first
///   *interpolates towards* across the access unit, so a matrix can move while
///   a block is decoded. Nothing here needs that yet; the bit that says so is
///   written clear, which is not optional — a decoder reads it whether or not
///   there is a matrix at all.
///
/// With the shift at zero and no dither the arithmetic is the other syntaxes'
/// exactly — accumulate, shift down by the fractional bits — so
/// [`crate::matrix`] describes one matrix for all three.
fn write_immersive_matrix_params<S: BitSink>(
    writer: &mut S,
    declared: &[&crate::matrix::Primitive],
    range: &crate::format::Substream,
) {
    if declared.is_empty() {
        writer.put(1, 0); // no matrix
        writer.put(1, 0); // and none of it moves
        return;
    }

    writer.put(1, 1); // a matrix follows
    writer.put(1, 1); // and it is described in full rather than re-coefficiented
    writer.put(4, declared.len() as u32 - 1);

    let sources = range.matrix_sources();
    for matrix in declared {
        writer.put(4, matrix.dest as u32);
        writer.put(4, matrix.frac_bits);
        // The shift a decoder reads as one less than the field, so one is
        // none. It scales the whole matrix — see [`crate::matrix::Primitive::wide`]
        // — and a matrix that moves never carries one, since the decoder would
        // scale its steps by it too.
        debug_assert!(matrix.shift == 0 || !matrix.moves());
        writer.put(3, matrix.shift + 1);
        writer.put(2, 0); // no bypassed least significant bits
        writer.put(4, 0); // and no dither
        writer.put(sources as u32, mask(matrix, sources));
    }
    for matrix in declared {
        for source in 0..sources {
            if matrix.coefficients[source] != 0 {
                writer.put_signed(matrix.frac_bits + 2, matrix.stored(source));
            }
        }
    }

    if declared.iter().any(|matrix| matrix.moves()) {
        write_interpolation(writer, declared, sources);
    } else {
        writer.put(1, 0); // nothing interpolates
    }
}

/// How far each matrix moves per access unit.
///
/// A decoder ramps the coefficient across the unit and then **adds the whole
/// step at the end of it**, so one statement tracks a trend for as long as the
/// matrix stands rather than for one unit. That is what makes it worth saying
/// at all: restating a matrix every unit costs more in description than the
/// tracking saves.
///
/// The width is per matrix and the coefficients follow the same mask as the
/// coefficients themselves — a source with no coefficient cannot have a step.
fn write_interpolation<S: BitSink>(
    writer: &mut S,
    declared: &[&crate::matrix::Primitive],
    sources: usize,
) {
    writer.put(1, 1); // something interpolates
    writer.put(1, 1); // and the steps are stated here
    writer.put(1, 1); // in full, rather than re-using the widths of a previous
    // statement

    let widths: Vec<u32> = declared
        .iter()
        .map(|matrix| step_width(matrix, sources))
        .collect();
    for width in &widths {
        writer.put(4, *width);
        writer.put(2, 0); // the steps are in the coefficients' own fractional
        // bits, so nothing further scales them
    }

    for (matrix, width) in declared.iter().zip(&widths) {
        if *width == 0 {
            // A matrix that does not move says so by its width, and states no
            // steps at all.
            continue;
        }
        for source in 0..sources {
            if matrix.coefficients[source] != 0 {
                writer.put_signed(width + 1, matrix.delta[source]);
            }
        }
    }
}

/// Bits one matrix needs for its steps, or zero for a matrix that stands
/// still.
fn step_width(matrix: &crate::matrix::Primitive, sources: usize) -> u32 {
    let widest = (0..sources)
        .filter(|source| matrix.coefficients[*source] != 0)
        .map(|source| matrix.delta[source].unsigned_abs())
        .max()
        .unwrap_or(0);
    if widest == 0 {
        return 0;
    }
    // A step is written signed in one bit more than this, so this is what has
    // to hold its magnitude.
    (32 - widest.leading_zeros()).min(15)
}

/// Which of a matrix's coefficients are there, one bit per source, lowest
/// channel in the lowest bit.
fn mask(matrix: &crate::matrix::Primitive, sources: usize) -> u32 {
    let mut mask = 0u32;
    for source in 0..sources {
        if matrix.coefficients[source] != 0 {
            mask |= 1 << source;
        }
    }
    mask
}

/// One filter's taps.
///
/// The shift is written with each filter and a decoder reads both — and then
/// uses only the first one's. The second is written to match rather than left
/// arbitrary, because "ignored" is a property of one decoder rather than of
/// the format.
pub(crate) fn write_filter_params<S: BitSink>(
    writer: &mut S,
    taps: &crate::filter::Taps,
    shift: u32,
) {
    writer.put(4, taps.order as u32);
    if taps.order == 0 {
        return;
    }
    writer.put(4, shift);
    writer.put(5, taps.coeff_bits);
    writer.put(3, taps.coeff_shift);
    for coefficient in &taps.coefficients[..taps.order] {
        writer.put_signed(taps.coeff_bits, coefficient >> taps.coeff_shift);
    }
    // A state block: the history the decoder is to hold, most recent value
    // first, each in `bits` and shifted up by `shift` as it is read. Only a
    // second filter carries one — see [`crate::filter::Taps::state`].
    match &taps.state {
        Some(state) => {
            writer.put(1, 1);
            writer.put(4, state.bits);
            writer.put(4, state.shift);
            for value in &state.values[..taps.order] {
                writer.put_signed(state.bits, value >> state.shift);
            }
        }
        None => writer.put(1, 0),
    }
}

/// The residuals.
///
/// Each is split by [`crate::huffman`] into a symbol priced by the channel's
/// codebook and a tail of raw bits. With no codebook there is no symbol and
/// the whole residual is the tail, which is what every channel did before the
/// codebooks landed.
fn write_block_data<S: BitSink>(
    writer: &mut S,
    unit: &Unit<'_>,
    block: &Block<'_>,
    range: &crate::format::Substream,
) {
    for frame in block.first..block.first + block.frames {
        for channel in range.first..=range.last {
            let coding = &block.codings[channel];
            let residual = unit.samples[channel][frame];
            let split = crate::huffman::split(coding.codebook, coding.huff_lsbs, 0, residual)
                .expect("the search only chooses a coding every residual fits");

            if let Some(symbol) = split.symbol {
                let (code, length) = crate::huffman::code(coding.codebook, symbol);
                writer.put(length, code);
            }
            if coding.huff_lsbs > 0 {
                writer.put(coding.huff_lsbs, split.lsbs);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The extra data block of the reference stream's first access unit is 78
    /// bytes: two that state its length, and 38 words after them. Its
    /// Evolution frame declares 73 bytes, which is the room after the
    /// Evolution header rather than the frame's own size — the frame is 579
    /// bits, its eight-bit protection field included, and the rest is padding.
    ///
    /// The arithmetic here reproduces those three numbers from the frame's bit
    /// count alone, which is what says the sizing rule is the reference's and
    /// not merely one that parses.
    #[test]
    fn the_block_sizes_the_way_the_reference_does() {
        let frame_bits = 579;
        let total_bits = (24 + frame_bits) as usize;
        let total_bits = total_bits.next_multiple_of(16);

        assert_eq!(total_bits / 16, 38, "words after the length");
        assert_eq!((total_bits - 24) / 8, 73, "the declared frame length");
        assert_eq!(2 + total_bits / 8, 78, "the whole block");
        assert_eq!(check_nibble(38), 0xb, "the reference's own check nibble");
    }

    /// The nibble is whatever makes the block's first two bytes fold to
    /// fifteen, which is the only thing distinguishing a block from padding.
    #[test]
    fn the_check_nibble_makes_the_first_two_bytes_fold_to_fifteen() {
        for words in [1u32, 2, 38, 255, 256, 4095] {
            let nibble = check_nibble(words);
            let high = ((nibble << 4) | (words >> 8)) as u8;
            let low = words as u8;
            let fold = high ^ low;
            assert_eq!((fold ^ (fold >> 4)) & 0xf, 0xf, "{words} words");
        }
    }
}
