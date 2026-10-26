# harlettizer

An open encoding engine for immersive audio, and the encoder counterpart to
[`harletty`](https://github.com/harletty/harletty-bridge). Where harletty
*decodes* immersive bitstreams into masters, harlettizer *encodes* masters into
bitstreams — today a **lossless TrueHD/MLP stream carrying object audio
metadata**, written from an object master and checked bit for bit through
decoders that share no code with it — and **IAMF**, the open format: the same
master rendered to a 7.1.4 bed that today's IAMF players decode, or carried as
IAMF v2.0 objects, coded as FLAC, LPCM or Opus, as a standalone stream or in
Matroska.

Pure Rust, headless, GPL-3.0-or-later.

```bash
harlettizer convert programme.atmos programme_adm.wav                  # master set -> ADM BW64
harlettizer encode  programme.atmos --out programme.thd --cluster 12   # the immersive bitstream
harlettizer iamf    programme.atmos --out programme.iamf               # a 7.1.4 bed as IAMF (.mka: in Matroska)
```

## What it does

Three commands, each taking a master file set (`.atmos` + `.atmos.audio` +
`.atmos.metadata`) or an ADM BW64 file.

| | |
|---|---|
| `convert` | One container to the other, either direction |
| `encode` | The programme as a TrueHD stream: its elements coded losslessly, its object metadata carried in the access units, its objects clustered into the twelve to sixteen elements a delivery stream holds, and — on request — real 2.0, 5.1 and 7.1 folds inside the stream for decoders that stop early |
| `iamf` | The programme as an IAMF sequence: by default rendered to a 7.1.4 bed under IAMF v1.1, what browsers and televisions decode; with `--objects`, its objects carried as IAMF v2.0 objects beside its bed. FLAC, LPCM or Opus; a standalone stream, or the one track of an `.mka`. See [`iamf`](#iamf) below |

**Everything an encode writes is decoded back and compared, not inspected.**
The test suite decodes every presentation of every stream it writes through a
reference decoder in the test tree and verifies the lossless check each restart
header carries. The development harness does the same through `harletty` (the
`truehd` crate) and, up to eight channels, through FFmpeg. On a real programme
the twelve elements come back with a largest difference of zero, and every
object comes back where the master put it, at the sample the master named.

The IAMF sequences are held to decoders the same way: FFmpeg and libFLAC
decode the bed to exactly the samples written, iamf-rs renders it to 7.1.4
and stereo as the stream says, and `harletty` decodes a lossless v2.0
sequence back to the master's own channels, bit for bit.

### Where it stands

- **Channel-based streams** — mono, stereo, 5.1 and 7.1, within a few per
  cent of FFmpeg's TrueHD encoder either way, measured on slices of a
  programme:

  | | mono | stereo | 5.1 | 7.1 |
  |---|---|---|---|---|
  | size, as a ratio of FFmpeg's | 0.96 | 0.94 | 0.97 | 1.06 |

- **Immersive streams** — nine to sixteen elements over four substreams, the
  object metadata read back by a real decoder as the programme it describes,
  and a real programme at about the size of its reference stream. With
  `--presentations` the stream carries genuine folds, about 8 % larger than
  without; see [docs/mlp.md](docs/mlp.md).
- **Speed** — about 3 ms per channel-second of audio at full effort on a
  desktop machine: four minutes of a twelve-element programme in 17 s on
  sixteen cores.
- **IAMF beds** — a 7.1.4 bed under the v1.1 simple profile, with its own
  FLAC encoder within a third of a per cent of `flac -8 --no-mid-side` (IAMF
  forbids mid/side), and the loudness stated on 7.1.4 and on stereo what
  FFmpeg's meter reads on a decoder's render. Opus at 96 kbit/s a channel by
  default, about 1.08 Mbit/s for 7.1.4.
- **IAMF objects** — up to twenty-eight channels: the bed as one element,
  every object its own element with its path as position blocks, a master with
  more objects than that folded by the same clustering `encode` uses; a
  re-voiced programme kept element for element (`--overlay`) or with its voices
  as a labelled dialogue element (`--voices-to-bed`). Few decoders read v2.0
  yet: the `harletty/iamf-rs` fork and `harletty` do; FFmpeg 9 does not.
- **AC-3** — 5.1 or stereo, rendered from the master and coded by a port of
  FFmpeg's encoder, coupling included, which it matches to 0.01 dB SNR at
  every rate tried; no dynamic range words yet. See [docs/ac3.md](docs/ac3.md).
- **DTS:X** — the standard profile, a fixed 7.1.4: rendered from the master,
  each height folded into the 7.1 below it, the 7.1 coded losslessly beside a
  5.1 core and the heights losslessly beside it — every channel comes back bit
  for bit through the `dca` decoder, the 7.1 through FFmpeg, which reports the
  DTS:X profile. Or DTS-HD Master Audio 7.1 or 5.1 alone, at 16 or 24 bits.
  Raw or in Matroska (`A_DTS`). Not yet played on a receiver. See
  [docs/dts.md](docs/dts.md).
- **E-AC-3** — one independent substream at any rate up to 1024 kb/s, with an
  Evolution frame carried where reference streams carry theirs: a reference
  stream's objects ride on a core coded here and come back as close as the
  core does; and objects of our own, from a master of up to fifteen
  elements (`--joc`). No dependent substream yet. See
  [docs/eac3.md](docs/eac3.md).
- **Joint object coding** — the payload read and written, every reference
  payload byte for byte ([docs/joc.md](docs/joc.md)); the parameters
  estimated through the decoder's synthesis — to a reference's own core and
  objects, or, from a master, to the 5.1 core a decoder plays of ours. A
  master with more than fifteen elements is refused until its objects can be
  clustered.
- **Not there yet** — AC-4 is in [PLAN.md](PLAN.md) and nothing of it is
  written. For IAMF: no AAC, no
  v1.1 bed beside the v2.0 objects for older decoders, no scalable layers, and
  no browser has been tried; [docs/iamf.md](docs/iamf.md#not-done) has the
  list.

## Building

```bash
cargo build --release -p hz-cli
./target/release/harlettizer --help
```

Rust 1.87 or later. Pure Rust: no C toolchain, no vendor SDK, no git
dependency. The one exception is opt-in: `--features opus` links the system's
libopus for `iamf --codec opus`; see [docs/iamf.md](docs/iamf.md#opus).

Each release also carries the built binary for Linux (x86-64), Windows
(x86-64) and macOS (Apple silicon), zipped with the licences it ships under.
Those binaries have Opus, with libopus linked statically: nothing to install
beside them.

## Usage

### `convert`

```bash
harlettizer convert programme.atmos programme_adm.wav   # master set -> ADM BW64
harlettizer convert programme_adm.wav programme.atmos   # ADM BW64 -> master set
```

The direction is taken from the input. `--presentation N` picks which
presentation of a master set to convert (default 0). Two things a master set
says have no place in ADM and are reported when they are dropped: its language,
and its frame rate, since ADM times are absolute.

### `encode`

```bash
harlettizer encode programme.atmos --out programme.thd --cluster 12 --fold-depth 18 --presentations
```

Thirty seconds of a programme, eleven objects and an LFE into twelve
elements, at the default depth and with no signing key configured:

```
  programme    11 objects and an LFE, 12 elements
  encoded      36106 access units, 1444240 samples
  metadata     285 payloads, one every 127 units
  protection   no key configured, so a constant stands in the field; see docs/encode.md
  fold         0.0000 mean over the presentations, 0.000 at its worst, as a fraction of the object's own gains
  weighed by   the plain power, what a meter reads
  noticed      0.00 % of the element-windows wobble past the blur, 0.00 % of what is playing sitting in one; 0.00°/s of movement the mix never asked for, against 0.00°/s it did — see hz_cluster::motion
  held by      a dead band: an element stays where it was unless moving pays 25 % of what staying costs
  steadied by  a hold over 0 blocks behind and 2 ahead
  floor        -102.6 dBFS at the playback level a dialnorm of -31 implies; 3.33 objects a block under it
  depth        elements rounded to 20 bits
  headroom     0 blocks where an element's coherent peak was bounded in the fit, 0 blocks where the limiter brought every element down, the mix peaking at 0.22 of full scale before it
  drc          from the measured curve, per presentation, restated every 128 units
  presentations the folds themselves, in every one of 283 restart intervals
  wrote        programme.thd
```

| Option | Default | What it does |
|---|---|---|
| `--out <FILE>` | required | Write the stream here |
| `--cluster <ELEMENTS>` | | Fold the master's objects into this many elements — twelve, fourteen or sixteen — rather than giving each object one of its own. Without it the stream is one element per object, which needs a master that has already been clustered. The LFE is a bed, takes the first element untouched, and is not clustered |
| `--fold-depth <BITS>` | `20` | Round the folded elements to this many bits. A mix is a sum over real weights, so its low bits are the noise of the multiplication, and a lossless coder carries them anyway; twenty is worth about a third of the stream against twenty-four and puts the rounding near −120 dBFS. Seventeen is the floor. Match it to the source master. Only means anything with `--cluster` |
| `--overlay <SOURCES>` | | For a programme being re-voiced: keep the master's elements exactly as they are — metadata element for element, samples copied where nothing is added — and pan its last this-many objects, the new sources, onto them. Guarded: an encode that would put a source too far from where it asked to be, or make a still voice wander, is refused. Mutually exclusive with `--cluster`; see [docs/encode.md](docs/encode.md#--overlay-keeping-a-scene-and-adding-to-it) |
| `--overlay-beds`, `--overlay-drift`, `--overlay-spread`, `--overlay-wobble`, `--overlay-level`, `--overlay-cost`, `--overlay-fallback`, `--overlay-spare`, `--overlay-report` | | The guards of `--overlay`, whether a source may take a spare element, and the per-block account of what it did; each is documented in `harlettizer encode --help` and [docs/encode.md](docs/encode.md) |
| `--fold-search <PASSES>` | `4` | Passes of the search that places each element where the fold's own cost is lowest. Zero declines it. Worth about a fifth of what the fold costs and a twentieth of the stream. Only means anything with `--cluster` |
| `--beds <RULE>` | `auto` | What a fold does with a bed channel other than the LFE: `pinned`, an element of its own at its speaker's place; `free`, an object of the bed class sharing the elements with the rest; or `auto`, pinned while every object can still have an element beside the bed and free otherwise. Only means anything with `--cluster`; without it a bed other than the LFE is refused |
| `--fold-hold <FRACTION>` | `0.25` | How much better somewhere else has to be, as a fraction of what staying costs, before an element moves there. It takes an element's wobble — leaving a place and coming back within two tenths of a second — from 3.63 % of the windows it is heard in to 0.84 % on a programme folded one element short, for an error that does not move. Nought turns it off. Only means anything with `--cluster` |
| `--smooth-behind <N>` | `0` | Blocks behind the one being placed whose energies steady it, beside the two ahead that are always taken. Two of them take better than a third off an element's out-and-back excursions on a real contested fold, at no cost in mean error; nought until a listening settles it. Only means anything with `--cluster` |
| `--dialnorm <DB>` | `-31` | The programme's dialnorm, which sets the floor under what an object has to carry to be heard at the playback level: full scale at 105 dB SPL less the gain a decoder applies to bring the dialogue to −31, and the threshold of hearing under that — −102.6 dBFS at the reference. Below it an object seeds nothing and sets no worst case. Only means anything with `--cluster` |
| `--headroom <RULE>` | `limit` | How the fold keeps its elements inside the codec's domain: `limit`, a limiter with one gain for every element, ahead of the peak; `bound`, the fit bounding an element's coherent peak, which costs error and does not take every clip; `both`; or `off`, which clamps and counts the clips as it did. Only means anything with `--cluster` |
| `--loudness <RULE>` | `flat` | How a block's power is weighed before it steers the fold: `flat`, the plain power a meter reads; `kweighted`, through the K filter of BS.1770; or `perceptual`, what each object adds to the loudness of the scene, band by band, with what is near it masking it. Flat is what every measurement in `docs/clustering.md` was made under, and the perceptual rule did not beat it on the metric; it is there to be listened to, at about a third more encoding time. Only means anything with `--cluster` |
| `--presentations` | off | Make the stream carry real folds: a decoder stopping after any substream is handed a 2.0, a 5.1 or a 7.1 rather than the leading elements. A full decode is unchanged and still lossless. Off by default because it cannot always be done — the cascade that builds the hierarchy amplifies what it is given and the codec's domain is twenty-four bits, so a programme mixed near full scale is written without one, and the summary says so |
| `--drc <MODE>` | `measured` | The dynamic range word each presentation states: `measured`, a gain in decibels, or `off`. Metadata only; a decoder asked for full range ignores it and the samples are unchanged |
| `--fast` | off | Search less hard: one second-filter design, decided at every other restart. About a quarter off the time of an encode for a fraction of a per cent of stream |
| `--frames <N>` | | Stop after this many frames of audio |
| `--mono-prefix <PREFIX>` | | Read a master set's waveforms from `<PREFIX>_<n>.wav`, one mono file each in the master's own order, rather than from the interleaved file its header names — what `harletty decode --mono-prefix` writes. For a master being made rather than delivered, where interleaving only to read the files back apart would copy the whole programme. The stream is byte-identical either way |
| `--progress` | off | One `progress <n>%` line on standard error each time the whole percent changes — at most a hundred and one lines however long the programme — for a caller driving a bar. Standard output is untouched and the stream is byte-identical either way |

What the summary says:

- **programme** — what the master holds and how many elements the stream will
  carry. Nine is the fewest the format's shape allows and sixteen the most; a
  smaller programme is padded with inactive elements the syntax has a flag for.
- **encoded** — how many access units and samples went into the stream.
- **metadata** — how many object audio metadata payloads went out. One goes out
  when some element's state changes, and otherwise every 128 access units.
- **protection** — whether each payload's frame is signed, and with which key;
  see [docs/encode.md](docs/encode.md).
- **fold** — what the clustering cost: how far every object lands from where
  the mix put it once folded into the elements, measured from the metadata
  alone over every presentation the stream will be played through. An encoder
  that folds a scene and does not say what it cost is asking to be trusted.
- **weighed by**, **noticed**, **held by**, **steadied by**, **floor** — how
  the fold weighed the blocks, how much the elements moved that the mix did
  not ask for, and the rules that kept them still; `--cluster` only.
- **depth** — how many bits the folded elements were rounded to.
- **headroom** — what kept the folded elements inside the codec's domain: the
  fit's bound, the limiter, and how loud the mix got before them.
- **clipped** — an element is a sum, and a sum of objects that agree is louder
  than any of them. Samples past the codec's twenty-four bits are clamped rather
  than wrapped, and counted here when there are any; the line is absent when
  there are none.
- **drc** — the dynamic range word each presentation states, or none.
- **presentations** — whether the stream carries the folds, in how many
  restart intervals, or why an interval could not.

`--overlay` adds its own lines: **overlay**, **untouched**, **beds** and
**carried by** say what was kept and where the sources went (see
[docs/encode.md](docs/encode.md)), and four guards say what it cost —
**drift**, how far the carriers put a source from where it asked to be;
**fell back**, how often a source was put on the nearest bed outright;
**unasked**, movement a carrier added under a source that did not move; and
**stranded**, a source audible while every element was silenced.

Two environment variables make the encoder talk, for diagnosis: `HZ_TIME=1`
times its phases and prints them at the end, and `HZ_FOLD=1` says at every
restart how the presentations were arranged and why a build was refused where
one was.

To look at what was written:

```bash
harletty decode programme.thd --presentation 3 --output-path programme_back   # the elements, as a master set
cargo xtask thd programme.thd                                                 # the stream's structure and integrity
cargo xtask ffmpeg-check programme.thd                                        # whether every player can open it
```

### What a stream carries

Four substreams, each a presentation a decoder may stop at: two channels, six,
eight, and the elements. Without `--presentations` the first three are the
leading elements; with it they are the folds themselves, and the last
substream says how to turn them back into elements. A restart header every 128
access units, the object metadata beside the audio, a dynamic range word per
presentation, and the high-resolution timing field that lets a decoder joining
mid-stream say where in the programme it is.
[docs/mlp-state.md](docs/mlp-state.md) lists every constant and decision in
force; [docs/mlp.md](docs/mlp.md) is the notebook of how they were arrived at.

### `iamf`

```bash
harlettizer iamf programme.atmos --out programme.iamf                   # a 7.1.4 bed, FLAC
harlettizer iamf programme.atmos --out programme.iamf --codec opus      # the same, Opus, for distribution
harlettizer iamf programme.atmos --out programme.mka --objects          # IAMF v2.0 objects, in Matroska
harlettizer iamf vf.atmos --out vf.iamf --objects --voices-to-bed 7     # a dub's voices as a dialogue element
```

Two ways to carry a programme, and the format's two generations:

- **A bed** (default) — the bed channels routed to their speakers and the
  objects panned on the room's cube into one 7.1.4 channel-based element,
  under the IAMF v1.1 simple profile, the one browsers, televisions and
  FFmpeg read today. The objects do not survive as objects.
- **Objects** (`--objects`) — IAMF v2.0: the master's bed as one
  channel-based element on the smallest IAMF layout that holds it, and every
  object an element of its own, its path through the room written as position
  parameter blocks (`--positions cart16`, `cart8` or `polar`), every temporal
  unit a key frame so a seek lands on the right place. Up to twenty-eight
  channels; a master with more objects than that is folded by the clustering
  `encode --cluster` uses (`--elements N` sets the most). For a re-voiced
  programme, `--overlay K` keeps the master's elements and pans its last K
  objects onto them, the computation `encode --overlay` makes, and
  `--voices-to-bed K` renders the last K into a dialogue element of their own
  — labelled, with ±12 dB a player may offer, and a loudness anchored on it.

Either way the mix presentation states the loudness measured on 7.1.4 and on
the stereo pair a decoder folds to. FLAC (default) and LPCM are lossless, at
16 or 24 bits (32 for LPCM); Opus (`--codec opus`, `--bitrate`, in kbit/s a
channel, 96 by default) needs a build with the `opus` feature, which the
release binaries have. An `--out` ending in `.mka` or `.mkv` writes the
sequence as the one `A_IAMF` track of a Matroska file, following the draft
mapping proposed to the CELLAR working group; anything else is a standalone
IAMF stream, which FFmpeg puts in MP4 with the line in
[docs/iamf.md](docs/iamf.md#into-mp4). The summary says what the programme
became, the loudness and peaks on both layouts, and any sample the render
clipped.

The options, what each mode was checked against and what is not done are in
[docs/iamf.md](docs/iamf.md).

To look at what was written:

```bash
harletty info programme.iamf                                      # what the sequence declares
harletty decode programme.iamf --output-path programme_back       # back to a master set: the bed, or the objects with their paths
mkvinfo programme.mka                                             # the Matroska track
```

### `ac3`

```bash
harlettizer ac3 programme.atmos --out programme.ac3                    # 5.1 at 448 kb/s
harlettizer ac3 programme.atmos --out programme.ac3 --layout 2.0       # stereo at 192 kb/s
harlettizer ac3 programme.atmos --out programme.ec3 --eac3 --bitrate 768   # E-AC-3, 5.1
harlettizer ac3 programme.atmos --out programme.ec3 --eac3 --joc       # E-AC-3, 5.1 and its objects
```

Renders the programme to 7.1 on the room's cube, folds it to 5.1 or stereo
the way the TrueHD presentations fold, and codes it as an AC-3 elementary
stream at `--bitrate` (32 to 640 kb/s), stating `--dialnorm`. The summary
reports the loudness measured on the render beside it. The encoder, how it
compares with FFmpeg's and what it does not do yet are in
[docs/ac3.md](docs/ac3.md).

### `dts`

```bash
harlettizer dts programme.atmos --out programme.mka                    # DTS-HD MA + DTS:X, 7.1.4, in Matroska
harlettizer dts programme.atmos --out programme.dts                    # the same, a raw elementary stream
harlettizer dts programme.atmos --out programme.mka --layout 5.1 --bits 16   # DTS-HD MA 5.1, 16-bit
```

Renders the programme to 7.1.4 on the room's cube, as `iamf` does, and writes
DTS-HD Master Audio with the DTS:X extension's standard profile: the 7.1 with
each height folded into its corner at −3 dB, lossless at 24 bits beside a 5.1
core, and the heights, lossless, beside it. A limiter keeps each corner with
its height under full scale. `--layout 7.1` or `5.1` writes DTS-HD Master
Audio alone, at 24 or 16 bits (`--bits`); a 5.1 is the 7.1 render folded as
`ac3` folds it. The output is Matroska (`A_DTS`) when `--out` ends in `.mka`
or `.mkv`, a raw elementary stream otherwise. 48 kHz only. How it is checked,
what limits it, and how to try it on a receiver are in
[docs/dts.md](docs/dts.md).

## Development

```bash
cargo build --all-targets
cargo test --all
cargo fmt --all --check
```

CI gates formatting, the build and the full test suite; clippy is advisory.
The test suite needs no external decoder and no external material.

`cargo xtask` is the development harness — measurements and checks that need
real material or an external decoder:

| Command | What it does |
|---|---|
| `mlp` | Encode a file losslessly and check a decoder gets it back exactly — through FFmpeg up to eight channels, and through `harletty`, found on the `PATH` or named by `HARLETTY` |
| `thd` | Read a TrueHD stream's structure and check its integrity |
| `ffmpeg-check` | Check streams are ones every player can open: each substream's matrices counted against what a decoder reads, and the whole stream decoded by FFmpeg, failing on any line it prints |
| `overlay-check` | The invariant an `--overlay` encode rests on, checked against another decoder's output (`truehdd`, on the `PATH`) rather than against this project's own reader |
| `cluster` | Fold a master set's objects into the elements a bitstream carries, and say what it cost |
| `bench` | Render a master and each rule's fold of it on 7.1.4 as blinded stimuli for a listening session, and score a filled-in sheet into a ranking |
| `drc` | What gain a shipped stream states, and when it changes it |
| `loudness`, `speech` | Measure, or report on the speech gate, to hold next to another implementation |
| `roundtrip`, `container`, `adm` | Read every master set under a directory and write it back; rewrite audio containers; report what in a BW64 file's ADM was not understood |
| `make-master` | Write a small synthetic master file set to convert against — pure tones, a broadband scene of a voice, a rumble and effects, or the same with the effects starting and stopping; with every Nth object snapped or confined to the front, for scenes with classes; over an LFE, a left channel and the LFE, or a 7.1.2 bed |
| `corpus` | Survey a directory of master sets and report what they can be used to test |

The workspace is `hz-core` (formats and geometry), `hz-io` (the two
containers), `hz-analysis` (loudness, speech, dynamic range), `hz-render`
(layouts, the measured fold model, and the panner the clustering metric judges
the immersive presentation on), `hz-cluster` (objects into elements), `hz-meta`
(object audio metadata), `hz-mlp` (the lossless encoder), `hz-iamf` (IAMF
sequences), `hz-programme` (what every writer does before its own format:
reading a master, its tracks and object paths, rendering, mixing a fold or an
overlay, coding samples) and `hz-cli`. The codec crates never depend on each
other, and `hz-programme` depends on none of them; only `hz-cli` knows all of
them. `hz-stream` is the other front end: a DTS stream in, a TrueHD stream
out, in memory and in real time, for a player; `hz-stream-ffi` is its C ABI
([docs/stream.md](docs/stream.md)).
See [CONTRIBUTING.md](CONTRIBUTING.md).

## Documentation

| | |
|---|---|
| [docs/encode.md](docs/encode.md) | The `encode` command: what a real programme did, the progress protocol, what `--fast` costs |
| [docs/joc.md](docs/joc.md) | Joint object coding: the payload layer, how it is held to the reference streams, and the one shape every reference stream codes its objects in |
| [docs/eac3.md](docs/eac3.md) | `ac3 --eac3`: the independent substream, where the objects go, how it compares with FFmpeg, and the graft of a reference stream's objects onto our core |
| [docs/dts.md](docs/dts.md) | The `dts` command: the standard DTS:X profile and the 5.1 and 7.1 beds, raw or in Matroska, how they are checked, the navigation limit, and how to try it on a receiver |
| [docs/stream.md](docs/stream.md) | `hz-stream` and `hz-stream-ffi`: the pipeline a player drives, its threads and buffers, the C ABI and how it is versioned, and what it does about keys |
| [docs/ac3.md](docs/ac3.md) | The `ac3` command: the render and fold, the encoder ported from FFmpeg, how it compares, and what is not done |
| [docs/iamf.md](docs/iamf.md) | The `iamf` command: the bed and the v2.0 objects, the render, the loudness, the FLAC encoder, Opus, Matroska and MP4, the options, what checked each, and what is not done |
| [docs/mlp-state.md](docs/mlp-state.md) | What the MLP encoder writes today — every constant and decision in force |
| [docs/mlp.md](docs/mlp.md) | The lossless encoder's notebook: what was tried, what it measured, what was put down |
| [docs/clustering.md](docs/clustering.md) | Clustering objects into elements, and the metric that judges a clustering |
| [docs/clustering-next.md](docs/clustering-next.md) | What the clustering still lacks, the public sources for each quality, and how each lever is to be measured |
| [docs/fold.md](docs/fold.md) | How a shipped stream folds an object onto a channel presentation, measured |
| [docs/objects.md](docs/objects.md) | Object audio metadata, and what a payload can and cannot say |
| [docs/drc.md](docs/drc.md) | Dynamic range metadata |
| [docs/loudness.md](docs/loudness.md), [docs/speech.md](docs/speech.md) | Loudness and true peak, and the speech gate behind `dialnorm` |
| [docs/adm.md](docs/adm.md), [docs/master-format.md](docs/master-format.md) | The two containers, as implemented and as observed |
| [docs/provenance.md](docs/provenance.md) | Every piece of code, table or constant that came from another project |
| [PLAN.md](PLAN.md) | The phased plan, and what each phase was judged on |

## Legal

**GPL-3.0-or-later.** That choice is what lets FFmpeg's LGPL-2.1 encoders be
ported here; what was taken from FFmpeg, from the `truehd` crate (Apache-2.0)
and from the EBU ADM Renderer (BSD-3-Clause) — and, for IAMF, the stereo
rendering table of iamf-rs and libiamf and the v2.0 object syntax read from
libiamf's test vectors (BSD-3-Clause-Clear), and the Matroska mapping drafted
in Omniphony (GPL-3.0) — is recorded file by file in
[docs/provenance.md](docs/provenance.md), and [LEGAL.md](LEGAL.md) says which
sources are open to this project and which are closed. libopus (BSD-3-Clause)
is linked, not ported, and only in a build with the `opus` feature; the
release binaries carry its licence beside the others.

**Patents.** The formats implemented here are covered by patents held by their
format owners. Writing and distributing an independent implementation is
lawful where this project is developed; *using* it may require a licence
depending on jurisdiction and use. This project makes no claim that using it
is patent-free, and follows the posture of comparable projects: ship the
source, let each user assess their own obligations.

**Trademarks.** Format names are used descriptively and only where necessary.
The output is compatible with decoders implementing the published
specifications; it is not certified or approved by any format owner, and
bit-identical output to any proprietary encoder is neither achievable nor the
target. Decoder interoperability and audible parity are.
