# hz-stream and hz-stream-ffi

A player that cannot decode DTS:X to its objects can hand the stream to
`hz-stream` and get TrueHD back: the lossless core of the DTS stream, carried
as the sixteen-element presentation, with the object metadata the DTS:X frame
states. Nothing is written to disk; the player pushes packets in and pulls
access units out.

```
push_dts(bytes, pts) ──► [ decode thread ] ──► [ encode thread ] ──► pull_truehd(&mut unit, &mut pts)
                          dca::HdDecoder        hz_mlp::Encoder
```

## What it does

- **Parses** the elementary stream into core frames and extension substreams,
  waiting for a substream cut by a packet boundary and resynchronising after
  junk (the size field is read from the header alone).
- **Decodes** with the `dca` crate from `harletty-bridge`, pinned by revision.
  The lossless layer is used where the frame has one; the core alone plays
  otherwise.
- **Cleans the bed** (`src/bed.rs`). A DTS:X lossless bed is the speakers with
  every extension waveform already folded in. The waveforms are carried again
  as elements of their own, so each frame's bed has the carried ones taken
  out first, with the decoder's own fold machinery: a stated fold is removed
  bit for bit, one the encoder panned from the object's position is
  recomputed, and any other is fitted to the audio (never adding energy to a
  speaker). A waveform whose fold stays unknown is not carried: its element is
  described as not there and silent, and its content stays in the bed. When
  there are nine waveforms the elements hold eight, and the one left out (it
  stays in the bed, folded) is a speaker object that is fully represented by
  its own fold, the one nearest the front, which is the centre object where
  there is one; with none, the last height.
- **Authors** one object audio metadata payload per frame (`src/oamd.rs`) and
  hands it to the encoder under `hz_meta::emdf::OAMD`, taking effect at the
  sample the frame starts at. The payload describes the sixteen elements in
  the order the encoder is given the planes: the low frequency channel, the
  seven bed channels at their speakers, then the carried waveforms at their
  stated positions (not there past the last one). The bed is described in
  every frame, with or without extension metadata. It covers both profiles
  the decoder tells apart: the standard one (four height waveforms) and the
  alternate ones (object records with a position each, then the four fixed
  heights). Positions are mapped from the stream's azimuth and elevation onto
  the cube the speakers sit in, anchored on the speaker azimuths (see the
  module documentation). The encoder writes the Evolution frame round the
  payload and signs it when it has a key; the payload is the bare object
  metadata, as the file encoder (`hz-cli`) passes it. What is shared with
  `hz-cli` is `hz_meta::oamd`, the payload syntax and its writer. Nothing
  else is signed.
- **Encodes** at the real-time effort of `hz-mlp` by default, with the
  encoder's decorrelation of the early channels off (see Presentations).
- **Times** every output unit from the sample counter the pushed timestamps are
  keyed on; a unit whose time is not known gets NaN, never a number that looks
  valid.
- **Recovers**: a panic in either thread costs the audio in flight and nothing
  else.

## Buffers

A frame's planes and a packet's bytes go round a short loop of reusable
buffers (`src/pool.rs`), so that a frame costs no allocation once the stream is
warm. Each buffer is owned by exactly one party at a time and moves by value
through the channels; there is no `unsafe` and nothing is borrowed across a
send.

## Presentations

The stream always states its four presentations. With `Options::folds` off
(the default) the narrow ones are selections of the bed: 2.0 is L R, 5.1 is
L R C LFE Ls Rs and 7.1 is the bed in that order, each channel exactly the
samples of its element; the sixteen-element presentation is the elements
themselves. A selection asks the encoder for nothing it can refuse. Left
unstated, the narrow presentations would be the leading elements, which start
with the low frequency channel (a 2.0 decoder would play LFE and L).

That exactness holds because the encoder is built with its decorrelation of
the early channels off (`Encoder::set_decorrelation(false)`). With it on, a
stream that states presentations has the rows of its early substreams
rewritten, and on correlated content, which a real bed is, the 2.0, 5.1 and
7.1 outputs differ from the channels they name by up to hundreds of least
significant bits; the sixteen elements stay exact either way. The cost of
leaving it off was measured on one real alternate-profile sample: 9964 bytes
more in 14.36 MB (0.07%). The proper fix is in `hz-mlp` (the decorrelation
must keep each early substream's stated output exact). A test encodes
correlated content and checks all four presentations against their planes.

With `Options::folds` on, the 2.0 and 5.1 are real folds of the bed (the usual
coefficients). The encoder can refuse a fold on an interval (0.2 to 6.3% of
the intervals of real content, measured), and for such an interval it falls
back to the selection above (`hz_mlp::Encoder::set_presentations_with_fallback`,
counted in `folds_fallen_back`), so a narrow decoder never loses the channels
at all, it only plays the unfolded ones for that interval. The fold arithmetic
itself is in `hz-mlp`; only the policy is here.

## The C ABI

`hz-stream-ffi` builds a `cdylib` and a `staticlib`; `include/hz_stream.h` is
the contract. `hz_stream_abi_version()` returns the version the header
declares as `HZ_STREAM_ABI_VERSION`; a host that loads the library at run time
compares it with its own before calling anything else.

## Keys

The Evolution frames have a protection field, a keyed digest. This project
ships no key, derives none and helps nobody obtain one (see
[../LEGAL.md](../LEGAL.md) section 4). A host may pass a key it was given by
its own user; it is forwarded to the encoder's `set_evolution_key` and used for
nothing else. Without one the field holds a constant that verifies against
nothing, so a receiver that checks it may refuse the stream or play it without
the object metadata. The key signs the object metadata and nothing else.

## Cores

On a part with fast and slow cores the encoder's worker threads can be
confined to the fast ones: by the host (`hz_stream_set_cpu_cores`), by
`HZ_STREAM_CORES` (`2,3,4,5` or `none`), or, on Linux, by the cores that report
the highest capacity when the cores are not alike.

## Timing

`cargo bench -p hz-stream --bench synthetic` times the pipeline on synthetic
sixteen-channel input at each effort. Measurements on real streams are not part
of the repository.
