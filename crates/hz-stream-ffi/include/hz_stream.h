/*
 *  hz-stream: real-time DTS to TrueHD conversion for a player
 *  SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef HZ_STREAM_H
#define HZ_STREAM_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * The version of this interface. It changes whenever this header changes in a
 * way that a host built against the old one would not survive. A host calls
 * hz_stream_abi_version() on the library it loaded, before anything else, and
 * refuses a library whose answer is not the HZ_STREAM_ABI_VERSION it was
 * built with.
 */
#define HZ_STREAM_ABI_VERSION 1u

uint32_t hz_stream_abi_version(void);

typedef struct hz_stream_context hz_stream_context;

/*!
 * \brief Create a new real-time conversion context.
 * \param sample_rate Output sample rate. Only 48000 Hz streams are encoded;
 *        a DTS stream at another rate makes push return -3 and carries
 *        silence of the frame's length, so the caller should check the
 *        stream rate before using the codec.
 * \param effort_level 0 = Realtime (the default: fixed-order prediction, no
 *        second filter or coding matrices); 1 = Fast; any higher value = Full.
 * \return Pointer to allocated context, or NULL on error.
 *
 * Timestamps are microseconds. All entry points return -4 if
 * an internal panic was caught.
 */
hz_stream_context* hz_stream_create(uint32_t sample_rate, uint32_t effort_level);

/*!
 * \brief Destroy a context and free all resources.
 */
void hz_stream_destroy(hz_stream_context* ctx);

/*!
 * \brief Confine the encoder's worker threads to the given cores (for
 *        instance the fast cores of a big.LITTLE part).
 *        Call before the first hz_stream_create; later calls have no effect.
 *        With count 0 the threads are left unconfined. Without this call the
 *        HZ_STREAM_CORES environment variable ("2,3,4,5" or "none") is
 *        read, and on Linux the cores that report the highest capacity are
 *        used when the cores are not all alike.
 * \return 0 if taken, 1 if too late or unsupported on this platform, -1 for
 *         a bad argument.
 */
int hz_stream_set_cpu_cores(const int32_t* cores, size_t count);

/*!
 * \brief Drop everything of the stream so far, including audio the encoder
 *        holds back, so nothing from before a seek is played afterwards.
 *        The key is kept.
 */
void hz_stream_reset(hz_stream_context* ctx);

/*
 * Keys. The metadata frames the stream carries have a protection field, a
 * keyed digest. This library contains no key, derives none and does not help
 * anyone obtain one; it only passes on a key the host was given by its own
 * user, to the encoder, which signs with it. Without one the field holds a
 * constant that verifies against nothing, and a receiver that checks the
 * field may refuse the stream or play it without the object metadata.
 */

/*!
 * \brief Hand over the key the Evolution frames that carry the object
 *        metadata are signed with (they are the only thing signed), as raw
 *        bytes. It is kept across hz_stream_reset() and wiped when the
 *        context is destroyed.
 * \return 0 on success, -1 for a null/empty argument.
 */
int hz_stream_set_key(hz_stream_context* ctx, const uint8_t* key_ptr, size_t key_len);

/*!
 * \brief Same, from hexadecimal text (no NUL needed). The text must be an
 *        even number of hex digits (1 to 128 bytes) with at most surrounding
 *        whitespace; nothing else is accepted and it is never used as raw
 *        bytes.
 * \return 0 on success, -1 for a null argument, -5 for a malformed key (the
 *         previous key is kept).
 */
int hz_stream_set_key_hex(hz_stream_context* ctx, const uint8_t* text_ptr, size_t text_len);

/*
 * Threading: a context runs a decode thread and an encode thread of its own.
 * push only hands the bytes over and pull only pops what is ready, so neither
 * does codec work on the caller's thread. A caller that stops pulling stalls
 * the encoder after about a second of audio and then makes push wait; keep
 * pulling after every push. reset and flush wait for both threads. All calls
 * on one context must come from one thread at a time.
 */

/*!
 * \brief Push raw DTS / DTS-HD / DTS:X elementary stream bytes.
 * \param pts Presentation time of the packet in microseconds. A negative,
 *        non-finite or implausibly large value (such as DVD_NOPTS_VALUE) means
 *        "none".
 * \return Number of bytes consumed, -1 for a bad argument, -3 if an earlier
 *         frame had a sample rate other than the one the context was created
 *         for (reported once, by the next push, since decoding is behind the
 *         call), -4 if the pipeline has stopped (a caught panic in a call, or a
 *         thread that cannot recover). A panic inside a pipeline thread is
 *         recovered from: the audio in flight is lost, the packet is still
 *         consumed, and the next push goes on. A packet is therefore never
 *         to be sent again after a non-negative or -3 return.
 */
int hz_stream_push_dts(hz_stream_context* ctx, const uint8_t* data, size_t len, double pts);

/*!
 * \brief Pull ONE TrueHD access unit (40 samples at 48 kHz).
 * \param out_pts Time of the audio in this unit in microseconds, or NaN if no
 *        timestamp was ever supplied for it. Never a sentinel number.
 * \return Length of the unit in bytes; 0 if none is ready; -2 if max_len is
 *         smaller than the unit (nothing is consumed; call again with a
 *         larger buffer); -1 for a bad argument.
 */
int hz_stream_pull_truehd(hz_stream_context* ctx, uint8_t* out_buf, size_t max_len, double* out_pts);

/*!
 * \brief End of stream: decode what is buffered, finish the last partial
 *        unit (padded, with the padding marked) and drain the encoder. Pull
 *        the units afterwards. Input pushed after a flush starts a new stream.
 * \return Bytes now queued for pulling, or negative on error.
 */
int hz_stream_flush(hz_stream_context* ctx);

#ifdef __cplusplus
}
#endif

#endif // HZ_STREAM_H
