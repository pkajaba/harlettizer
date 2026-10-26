//! MLP and TrueHD lossless encoding.
//!
//! The point of a lossless codec is that it admits a decisive test: either
//! `decode(encode(x)) == x` bit for bit, or the encoder is wrong. Nothing here
//! is judged by listening or by inspection.
//!
//! ```no_run
//! use hz_mlp::{Config, Encoder, SampleBits};
//!
//! let mut encoder = Encoder::new(Config {
//!     sample_rate: 48_000,
//!     channels: 2,
//!     bits: SampleBits::TwentyFour,
//! })?;
//!
//! let unit = vec![0i32; encoder.frame_size() * 2];
//! let bytes = encoder.push(&unit);
//! # Ok::<(), hz_mlp::Unsupported>(())
//! ```

mod affinity;
pub mod arrange;
pub mod crc;
pub mod encoder;
pub mod filter;
pub mod format;
mod frame;
pub mod hierarchy;
pub mod hires;
pub mod huffman;
pub mod lpc;
pub mod matrix;
mod model;
pub mod protection;
mod rate;
pub mod reader;

pub use affinity::{confine_current_thread, set_pool_cores};
pub use encoder::Encoder;
pub use filter::Effort;
pub use format::{Config, MAX_CHANNELS, SampleBits, Unsupported};
pub use hz_core::bits::BitWriter;
