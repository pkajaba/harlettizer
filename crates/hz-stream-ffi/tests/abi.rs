//! The C ABI end to end, on streams the test writes itself.

use hz_dca::encoder::{Config, Encoder, FRAME_SAMPLES, Layout};
use hz_stream_ffi::*;

/// Ten core frames of a 5.1 tone.
fn frames() -> Vec<Vec<u8>> {
    let mut encoder = Encoder::new(Config::reference_five_one()).expect("a core encoder");
    let channels = Layout::FivePointOne.channels();
    (0..10)
        .map(|frame| {
            let pcm: Vec<i32> = (0..FRAME_SAMPLES * channels)
                .map(|i| {
                    let (n, c) = (i / channels, i % channels);
                    let t = (frame * FRAME_SAMPLES + n) as f64 / 48000.0;
                    ((t * (220.0 + 110.0 * c as f64) * std::f64::consts::TAU).sin()
                        * 0.25
                        * 8_388_607.0) as i32
                        & !0xff
                })
                .collect();
            let mut bytes = Vec::new();
            encoder.encode(&pcm, &mut bytes);
            bytes
        })
        .collect()
}

#[test]
fn the_version_is_the_headers() {
    assert_eq!(hz_stream_abi_version(), HZ_STREAM_ABI_VERSION);
    let header = include_str!("../include/hz_stream.h");
    assert!(header.contains(&format!(
        "#define HZ_STREAM_ABI_VERSION {HZ_STREAM_ABI_VERSION}u"
    )));
}

#[test]
fn every_entry_point_refuses_a_null_context() {
    unsafe {
        let mut buf = [0u8; 16];
        let mut pts = 0.0;
        let key = [1u8; 4];
        assert_eq!(
            hz_stream_push_dts(std::ptr::null_mut(), key.as_ptr(), key.len(), 0.0),
            -1
        );
        assert_eq!(
            hz_stream_pull_truehd(std::ptr::null_mut(), buf.as_mut_ptr(), buf.len(), &mut pts),
            -1
        );
        assert_eq!(hz_stream_flush(std::ptr::null_mut()), -1);
        assert_eq!(
            hz_stream_set_key(std::ptr::null_mut(), key.as_ptr(), key.len()),
            -1
        );
        assert_eq!(
            hz_stream_set_key_hex(std::ptr::null_mut(), key.as_ptr(), key.len()),
            -1
        );
        hz_stream_reset(std::ptr::null_mut());
        hz_stream_destroy(std::ptr::null_mut());
    }
}

#[test]
fn lifecycle_and_keys() {
    unsafe {
        let ctx = hz_stream_create(48000, 0);
        assert!(!ctx.is_null());

        let key = [0x5au8; 32];
        assert_eq!(hz_stream_set_key(ctx, key.as_ptr(), key.len()), 0);
        assert_eq!(hz_stream_set_key(ctx, key.as_ptr(), 0), -1);

        let good = "00112233445566778899aabbccddeeff";
        assert_eq!(hz_stream_set_key_hex(ctx, good.as_ptr(), good.len()), 0);
        for bad in ["0x00112233", "001", "zz", "", "00 11"] {
            assert_eq!(
                hz_stream_set_key_hex(ctx, bad.as_ptr(), bad.len()),
                -5,
                "{bad:?}"
            );
        }

        // Nothing pushed: nothing to flush, nothing to pull.
        assert_eq!(hz_stream_flush(ctx), 0);
        let mut buf = [0u8; 4096];
        let mut pts = 0.0;
        assert_eq!(
            hz_stream_pull_truehd(ctx, buf.as_mut_ptr(), buf.len(), &mut pts),
            0
        );
        hz_stream_reset(ctx);
        hz_stream_destroy(ctx);
    }
}

#[test]
fn the_cores_are_taken_once_or_refused() {
    unsafe {
        let list = [2i32, 3];
        assert_eq!(hz_stream_set_cpu_cores(std::ptr::null(), 2), -1);
        // Whether a first call is in time depends on the test that ran before;
        // the answers are the two the header names.
        let answer = hz_stream_set_cpu_cores(list.as_ptr(), list.len());
        assert!(answer == 0 || answer == 1, "{answer}");
    }
}

#[test]
fn a_stream_pushed_through_the_c_abi_comes_out_in_whole_units() {
    unsafe {
        let frames = frames();
        let ctx = hz_stream_create(48000, 0);
        assert!(!ctx.is_null());
        let mut out = vec![0u8; 65536];
        let mut pts = 0.0f64;
        let (mut units, mut bytes) = (0usize, 0usize);
        let mut pull = |units: &mut usize, bytes: &mut usize| loop {
            let n = hz_stream_pull_truehd(ctx, out.as_mut_ptr(), out.len(), &mut pts);
            assert!(n >= 0);
            if n == 0 {
                break;
            }
            assert!(pts.is_finite());
            *units += 1;
            *bytes += n as usize;
        };
        let rounds = 6; // 60 frames, 30720 samples
        for round in 0..rounds {
            for (i, frame) in frames.iter().enumerate() {
                let at = ((round * 10 + i) * FRAME_SAMPLES) as f64 * 1.0e6 / 48000.0;
                assert_eq!(
                    hz_stream_push_dts(ctx, frame.as_ptr(), frame.len(), at),
                    frame.len() as i32
                );
            }
            pull(&mut units, &mut bytes);
        }
        assert!(hz_stream_flush(ctx) > 0);
        pull(&mut units, &mut bytes);
        assert_eq!(units, (rounds * 10 * FRAME_SAMPLES).div_ceil(40));
        assert!(bytes > units * 4);
        hz_stream_destroy(ctx);
    }
}
