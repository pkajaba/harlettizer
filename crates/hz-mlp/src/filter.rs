// SPDX-License-Identifier: GPL-3.0-or-later
//
// Carries material ported from FFmpeg, which is licensed under the GNU Lesser
// General Public License, version 2.1 or later, and is used here under the
// GPL-3.0-or-later of this project as that licence permits:
//   libavcodec/mlpdsp.c — Copyright (c) 2007-2008 Ian Caulfield, 2009 Ramiro Polla
// What was taken and what was changed is recorded in docs/provenance.md.

//! The format's prediction filter, and choosing one.
//!
//! # The arithmetic is the contract
//!
//! A lossless predictor is only lossless if the encoder and the decoder make
//! *exactly* the same guess, so the guess cannot be made in floating point and
//! cannot be made in a different order. The decoder computes, per sample:
//!
//! ```text
//! accum    = Σ a[k]·signal[k] + Σ b[j]·residual[j]      (64-bit)
//! accum  >>= shift                                      (arithmetic)
//! sample   = accum + residual
//! ```
//!
//! and then pushes `sample` onto the signal history and `sample - accum` — the
//! residual — onto the residual history. This module does the same operations
//! in the same widths in the same order, and subtracts. [`lpc`](crate::lpc)
//! may use all the floating point it likes to *decide* the coefficients; once
//! they are integers, nothing here is approximate.
//!
//! # Two filters, one shift
//!
//! The `a` taps predict the next sample from past samples; the `b` taps
//! predict it from past prediction errors. Together they are an ARMA
//! predictor, and its error filter is `(1 − A)/(1 + B)`: the first filter
//! places zeroes, the second places poles. Poles are what let a sharp
//! resonance be cancelled by coefficients coarse enough to fit in the fields
//! the format provides — which is why a pure tone, which an eight-tap
//! all-zero filter handles badly, is what the second filter is for.
//!
//! Both use **the same shift**, and it is the first filter's: the decoder
//! reads a shift for each and uses only one. So the second filter's precision
//! is not its own to choose.
//!
//! **The second filter is a fixed design, not a fit.** Fitting `B` to the
//! residuals the first filter leaves finds nothing: after an eight-tap fit
//! what is left is the quantisation error of the coefficients, which is white,
//! and no model of past residuals predicts white noise. What `B` is good for
//! is a pair of poles that lifts the steep low-pass roll-off programme audio
//! has, which the first filter's zeros model badly — so [`SECOND_FILTERS`]
//! holds four designs, [`pick_second`] chooses between them once a restart
//! interval by coding in each of them the interval it is *about to meet*, and
//! the block fits its first filter to what the chosen one leaves. FFmpeg's
//! encoder does not use this filter at all.
//!
//! # The history crosses blocks, and stops at a restart
//!
//! Within a restart interval a decoder carries the last eight reconstructed
//! samples and the last four residuals across blocks and across access units,
//! whatever the orders were, and this encoder carries both for the same
//! reason. A **restart header wipes them**, along with the coefficients and
//! the orders — which is what makes a restart a point a decoder can join the
//! stream at, and would be pointless otherwise.
//!
//! That last sentence was wrong here for a while, and it cost a stream that
//! FFmpeg's decoder reconstructed exactly and `truehdd` did not: FFmpeg leaves
//! its state buffers alone across a restart, so an encoder that carried its
//! history over agreed with it and with nothing else. See
//! [`crate::encoder`].

use crate::huffman;
use crate::lpc::{self, MAX_ORDER};

/// The last samples a filter predicts from, most recent first.
pub type History = [i32; MAX_ORDER];

/// How many past samples the fit is allowed to look at.
///
/// The filter only ever *applies* to the block in hand, but choosing it from
/// forty-eight samples is choosing it from very little: an eight-tap fit has
/// eight unknowns, and a resonator's predictor is sensitive to the estimate in
/// a way that shows up directly in the residual. Measured on a pure tone, a
/// fit over the block alone left 13.4 bits a sample where the coefficient
/// precision alone should have allowed far fewer.
///
/// Past samples only. Looking forward would mean buffering, and the encoder
/// writes each unit as it arrives. Two hundred and fifty-six was where the
/// gain stopped while the fit was made on a pure tone; measured again over
/// the fixture set once the encoder held whole intervals, five hundred and
/// twelve is 0.26 to 0.40 % smaller on every programme fixture and neutral on
/// the synthetic ones, a thousand and twenty-four a little better still on
/// a fold-free programme and a little worse on one carrying folds, and four
/// thousand clearly worse on both.
pub const FIT_CONTEXT: usize = 512;

/// The most taps the second filter carries.
pub const MAX_IIR_ORDER: usize = 4;

/// One filter's taps, as the stream carries them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Taps {
    pub order: usize,
    pub coefficients: [i32; MAX_ORDER],
    /// Width of each coefficient in the stream. One to sixteen.
    pub coeff_bits: u32,
    /// How far each coefficient is shifted up after being read.
    pub coeff_shift: u32,
    /// A history to install in the decoder along with the taps. Only the
    /// second filter may carry one — both decoders refuse it on the first —
    /// and it is what lets a second filter start cleanly: its taps run over
    /// past *residuals*, and the residuals the decoder holds were made by
    /// whatever filter came before, which with poles near the unit circle
    /// poisons a whole block. A shipped stream sends one with every second
    /// filter it starts.
    pub state: Option<FilterState>,
}

/// A filter history as the stream carries it: `order` values, most recent
/// first, each written in `bits` and shifted up by `shift` when read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FilterState {
    pub values: [i32; MAX_ORDER],
    pub bits: u32,
    pub shift: u32,
}

/// The most bits one state value can be written in: the field is four bits.
const MAX_STATE_BITS: u32 = 15;

impl Taps {
    /// The bits these taps cost to describe.
    ///
    /// A closed form standing in for writing them into a
    /// [`BitCounter`](hz_core::bits::BitCounter), because this is called once
    /// per candidate per block and the counter walks every field. The two are
    /// held to agree by `the_closed_form_agrees_with_the_counter` below; a
    /// closed form nothing checks is a number that drifts from the writer.
    pub(crate) fn cost_bits(&self) -> usize {
        if self.order == 0 {
            4
        } else {
            // order, shift, coeff_bits, coeff_shift, the coefficients, and the
            // bit that says whether a state block follows — then the block.
            4 + 4
                + 5
                + 3
                + self.order * self.coeff_bits as usize
                + 1
                + self
                    .state
                    .map_or(0, |state| 4 + 4 + self.order * state.bits as usize)
        }
    }

    /// Whether these are the same taps: what a decoder keeps between blocks,
    /// which a history to install is not part of.
    pub fn same_taps(&self, other: &Taps) -> bool {
        self.order == other.order
            && self.coefficients[..self.order] == other.coefficients[..other.order]
            && self.coeff_bits == other.coeff_bits
            && self.coeff_shift == other.coeff_shift
    }
}

/// A channel's filter pair for one block, as the stream carries it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Filter {
    /// Taps over past samples.
    pub fir: Taps,
    /// Taps over past residuals.
    pub iir: Taps,
    /// How far right to shift the accumulator, shared by both. Four bits, and
    /// the format's decoders require it to be at least eight.
    pub shift: u32,
}

/// What a decoder carries between blocks: the samples it reconstructed and the
/// residuals it read, most recent first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct State {
    pub signal: [i32; MAX_ORDER],
    pub residual: [i32; MAX_IIR_ORDER],
}

impl Filter {
    /// The bits this filter costs to describe.
    pub fn cost_bits(&self) -> usize {
        self.fir.cost_bits() + self.iir.cost_bits()
    }

    /// What the decoder will predict for the next sample.
    #[inline]
    fn predict(&self, state: &State) -> i64 {
        let mut accumulator = 0i64;
        for (coefficient, past) in self.fir.coefficients[..self.fir.order]
            .iter()
            .zip(&state.signal)
        {
            accumulator += i64::from(*coefficient) * i64::from(*past);
        }
        for (coefficient, past) in self.iir.coefficients[..self.iir.order]
            .iter()
            .zip(&state.residual)
        {
            accumulator += i64::from(*coefficient) * i64::from(*past);
        }
        accumulator >> self.shift
    }
}

/// How a channel's block is coded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Coding {
    pub filter: Filter,
    /// Which entropy codebook prices the high bits, or zero for none.
    pub codebook: u8,
    /// Low bits written raw under the codebook. The field is five bits wide.
    pub huff_lsbs: u32,
}

/// Second filters, tried as they are.
///
/// The format's second filter runs over past residuals, so the pair is a
/// pole-zero whitening filter, `(1 − A(z)) / (1 + B(z))`. Fitting `B` to the
/// residuals the first filter leaves finds nothing — after eight taps what is
/// left is white, and that was measured and closed. What `B` is good for is
/// not modelling those residuals: it is a pair of poles near Nyquist that
/// lifts the steep low-pass roll-off programme audio has, which the first
/// filter's zeros alone model badly. A shipped stream carries almost the same
/// second filter on every channel and in every block — about `[1.72, 0.77]`
/// at order two, `0.83` at order one — which says it is a fixed design there,
/// not a fit. So these are fixed here too: the first filter is fitted to what
/// each leaves, and the block picks by exact cost, as it does between orders.
/// The set is the best three of a grid measured over three fixtures: 0.19 bit
/// a sample off the residuals, where the shipped stream's own pair scored
/// 0.18 on its own.
const SECOND_FILTERS: [&[f64]; 4] = [&[0.83], &[1.5, 0.6], &[1.8, 0.85], &[1.72, 0.77]];

/// How much of a channel's coded past each [`crate::encoder`] history keeps:
/// an interval or so, which is the context every filter decision in the next
/// one starts from.
pub const SECOND_CONTEXT: usize = 2048;

/// How many blocks of the interval ahead the second filter is judged on, and
/// the margin it has to win by, in hundredths — none: measured, any margin
/// left programme channels without a pair that paid.
pub(crate) const TRIAL_BLOCKS: usize = 44;
const TRIAL_MARGIN: usize = 0;

/// The design [`Effort::Fast`] tries: the shipped stream's own pair, which is
/// the one taken most often when all four are offered. Worth 0.25 % of a
/// programme at that effort over the order-two design that used to stand here.
const FAST_DESIGN: usize = 3;

/// How hard the encoder searches.
///
/// The difference is in the second filter's decision. At every restart,
/// `Full` codes each channel's recent past again in every mode and keeps the
/// cheapest; `Fast` tries only the design that pays most often, and only at
/// every other restart. Measured on the fixtures, `Fast` gives up 0.1 to
/// 0.2 % of a programme stream and half a per cent of a synthetic one, for
/// about half the encoding time.
///
/// `Realtime` is for an encoder that has to keep up with a live source on a
/// small core: no second filter, no coding matrices, and one first filter of a
/// fixed order per block, fitted and priced against the plain coding and the
/// filter already in force. See [`REALTIME_ORDER`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Effort {
    #[default]
    Full,
    Fast,
    Realtime,
}

/// The one filter order [`Effort::Realtime`] fits.
///
/// Chosen by measuring the fixtures (tonal, programme-like and noise) at
/// orders two to eight: past four the stream shrank by under a per cent for
/// the cost of a longer residual pass per block.
pub const REALTIME_ORDER: usize = 4;

impl Effort {
    /// The only first-filter order this effort fits, where it is fixed.
    pub(crate) fn fixed_order(self) -> Option<usize> {
        (self == Effort::Realtime).then_some(REALTIME_ORDER)
    }
}

/// Which second filter a channel should carry through the coming interval,
/// judged on the interval just coded — by coding it again.
///
/// A second filter is a design that stays on for as long as the signal keeps
/// its roll-off, so it is decided where the matrices are, once per restart
/// interval, and not block by block: a block-by-block search never entered
/// it on a quiet channel, where its whole description is a third of a block
/// and its gain a few bits, and when made to it churned between designs
/// instead. No model of the residuals' width will do as the judge, either
/// way: a flat fit's width promised gains on quiet channels that coding them
/// turned into losses, and a windowed fit's found a twelfth of what a
/// programme channel actually saves. So the judge is the block search itself,
/// with its exact costs, run over the last blocks of the channel's past in
/// every mode — and the cheapest mode wins by a margin, or none does. Returns
/// an index into the designs, or `None` for no second filter.
///
/// `restart` counts the restarts so far, and `current` is what the channel
/// carries now: at [`Effort::Fast`] only one design is tried, and only at
/// every other restart, the current one being kept in between.
pub fn pick_second(
    history: &[i32],
    current: Option<usize>,
    effort: Effort,
    restart: u64,
) -> Option<usize> {
    if effort == Effort::Realtime {
        return None;
    }
    if history.len() < TRIAL_BLOCKS * crate::format::MAX_BLOCK.min(40) + FIT_CONTEXT {
        return None;
    }
    let candidates: &[usize] = match effort {
        Effort::Full => &[0, 1, 2, 3],
        Effort::Fast if restart % 2 == 1 => return current,
        Effort::Fast => &[FAST_DESIGN],
        Effort::Realtime => return None,
    };
    // The five trials are independent — each codes the same history in one
    // mode — and there are only twelve channels to spread over thirty-two
    // threads, so this is where the rest of the machine gets used. Reduced in
    // candidate order afterwards, with the same strict `<`, so the answer is
    // the one the sequential loop gives.
    #[cfg(feature = "parallel")]
    let costs: Vec<usize> = {
        use rayon::prelude::*;
        std::iter::once(None)
            .chain(candidates.iter().map(|index| Some(*index)))
            .collect::<Vec<_>>()
            .par_iter()
            .map(|mode| trial(history, *mode))
            .collect()
    };
    #[cfg(not(feature = "parallel"))]
    let costs: Vec<usize> = std::iter::once(None)
        .chain(candidates.iter().map(|index| Some(*index)))
        .map(|mode| trial(history, mode))
        .collect();

    let mut best = None;
    let mut best_cost = costs[0];
    for (&index, &cost) in candidates.iter().zip(&costs[1..]) {
        if cost
            + if TRIAL_MARGIN == 0 {
                0
            } else {
                cost / TRIAL_MARGIN
            }
            < best_cost
        {
            best_cost = cost;
            best = Some(index);
        }
    }
    best
}

/// What the last blocks of `history` cost coded in one mode, as the block
/// search codes them: each from the state the one before left, the filter
/// kept between them.
fn trial(history: &[i32], second: Option<usize>) -> usize {
    let block = crate::format::MAX_BLOCK.min(40);
    // Arrays rather than a vector per block: this runs over forty-odd blocks
    // per channel per restart, in every mode.
    let mut residual_buffer = [0i32; crate::format::MAX_BLOCK];
    let residuals = &mut residual_buffer[..block];
    let mut carried = [0i32; MAX_IIR_ORDER];
    let mut carried_len = 0usize;
    let mut in_force = Filter::default();
    let mut total = 0usize;
    let mut at = history.len() - TRIAL_BLOCKS * block;
    while at + block <= history.len() {
        let state = state_from(&history[..at], &carried[..carried_len]);
        let (coding, cost) = choose_with(
            second,
            &history[..at],
            state,
            &history[at..at + block],
            residuals,
            &in_force,
        );
        total = total.saturating_add(cost);
        in_force = Filter {
            iir: Taps {
                state: None,
                ..coding.filter.iir
            },
            ..coding.filter
        };
        let tail = residuals.len().saturating_sub(MAX_IIR_ORDER);
        carried_len = residuals.len() - tail;
        carried[..carried_len].copy_from_slice(&residuals[tail..]);
        at += block;
    }
    total
}

/// The widest residual the uncompressed coding can carry.
///
/// A residual is written in `huff_lsbs` bits and the field holds five, but a
/// residual wider than the samples themselves means the filter made things
/// worse, and order zero is always available.
const MAX_HUFF_LSBS: u32 = 24;

/// Choose a filter for one block and write out its residuals.
///
/// Every order is tried and the residuals actually computed, rather than
/// estimated from the prediction error the fit reports. The fit's error is a
/// statistic about a windowed, real-valued model; what this coding pays for is
/// the *largest* residual in the block, because that is what sets the field
/// width every sample in it is written at. Those are different questions, and
/// on a forty-sample block they have different answers often enough to matter.
///
/// `state` is what the decoder holds coming in, and `context` is what has
/// already been encoded, in time order, most recent last, for the fit to be
/// made over. **They are two different things and a restart header separates
/// them**: it wipes the decoder's state to zero, while the signal's history is
/// still the best guide to what its next forty samples look like. Passing the
/// history as the state as well produced a stream one decoder reconstructed
/// exactly and the other did not — see [`crate::encoder`].
///
/// Returns the coding chosen and what it costs in bits, filter description
/// included. `residuals` is filled for that coding. The cost is returned
/// because the caller has a choice this function cannot see — whether to
/// rematrix this channel against another first — and comparing those needs a
/// number, not a verdict.
pub fn choose(
    context: &[i32],
    state: State,
    samples: &[i32],
    residuals: &mut [i32],
    in_force: &Filter,
) -> (Coding, usize) {
    choose_with(None, context, state, samples, residuals, in_force)
}

/// As [`choose`], with the second filter the channel carries through this
/// interval already decided — see [`pick_second`]. With one decided, every
/// candidate is a pair built on it and the block chooses only the first
/// filter; with none, no pair is offered at all.
pub fn choose_with(
    second: Option<usize>,
    context: &[i32],
    state: State,
    samples: &[i32],
    residuals: &mut [i32],
    in_force: &Filter,
) -> (Coding, usize) {
    choose_fixed(second, None, context, state, samples, residuals, in_force)
}

/// As [`choose_with`], with the first filter's order fixed where `order` says:
/// one fit, one candidate, instead of every order up to the format's largest.
pub fn choose_fixed(
    second: Option<usize>,
    order: Option<usize>,
    context: &[i32],
    state: State,
    samples: &[i32],
    residuals: &mut [i32],
    in_force: &Filter,
) -> (Coding, usize) {
    debug_assert_eq!(samples.len(), residuals.len());
    let past = context;

    // Order zero: the sample is its own residual. Always available, and the
    // yardstick every filter has to beat.
    let (plain_coding, plain_bits) = plain(samples, residuals);
    let Some((codebook, huff_lsbs, bits)) = (plain_bits != usize::MAX).then_some((
        plain_coding.codebook,
        plain_coding.huff_lsbs,
        plain_bits,
    )) else {
        // Nothing can code these, which cannot happen for a 24-bit sample and
        // would be a bug rather than an input. Fall back to raw and let the
        // writer's own check catch it.
        return (
            Coding {
                filter: Filter::default(),
                codebook: huffman::NONE,
                huff_lsbs: MAX_HUFF_LSBS,
            },
            samples.len() * MAX_HUFF_LSBS as usize,
        );
    };
    let mut best = Coding {
        filter: Filter::default(),
        codebook,
        huff_lsbs,
    };
    let mut best_cost = bits + best.filter.cost_bits();

    // The filter already in force is free: the block can simply not mention
    // it. Nothing else here is, so a filter that keeps earning its keep is
    // compared against the alternatives on its residuals alone — which is the
    // whole reason a deep filter can pay for itself over a forty-sample block
    // when describing one afresh never could.
    //
    // The two candidate buffers are the block's own length, and a block is
    // never longer than the format's largest — so they are arrays rather than
    // two heap allocations per channel per block.
    let mut kept_buffer = [0i32; crate::format::MAX_BLOCK];
    let kept = &mut kept_buffer[..samples.len()];
    if (in_force.fir.order > 0 || in_force.iir.order > 0)
        && fill_residuals(in_force, &state, samples, kept) <= MAX_HUFF_LSBS
        && let Some((codebook, huff_lsbs, bits)) = best_entropy(kept)
        && bits < best_cost
    {
        best_cost = bits;
        best = Coding {
            filter: *in_force,
            codebook,
            huff_lsbs,
        };
        residuals.copy_from_slice(kept);
    }

    // The fit sees what came before as well as the block itself.
    let mut context = [0.0f64; FIT_CONTEXT + crate::format::MAX_BLOCK];
    let carried = past.len().min(FIT_CONTEXT);
    let total = carried + samples.len();
    if total > context.len() {
        fill_residuals(&best.filter, &state, samples, residuals);
        return (best, best_cost);
    }
    for (slot, sample) in context[..carried]
        .iter_mut()
        .zip(&past[past.len() - carried..])
    {
        *slot = f64::from(*sample);
    }
    for (slot, sample) in context[carried..total].iter_mut().zip(samples) {
        *slot = f64::from(*sample);
    }

    let mut scratch_buffer = [0i32; crate::format::MAX_BLOCK];
    let scratch = &mut scratch_buffer[..samples.len()];
    // Only when no second filter is in force: with one, every candidate is a
    // pair built on it and this fit is never looked at. Computing it anyway
    // was a Levinson-Durbin over the whole context per block, thrown away.
    if second.is_none() {
        let top = order.unwrap_or(MAX_ORDER);
        let fitted = lpc::fit(&context[..total], top);
        for order in order.unwrap_or(1)..=fitted.orders {
            let Some(quantised) = lpc::quantise(&fitted.coefficients[order - 1], order) else {
                continue;
            };
            let filter = Filter {
                fir: describe(&quantised.coefficients, order),
                iir: Taps::default(),
                shift: quantised.shift,
            };
            if fill_residuals(&filter, &state, samples, scratch) > MAX_HUFF_LSBS {
                continue;
            }
            consider(
                &filter,
                scratch,
                &mut best,
                &mut best_cost,
                residuals,
                filter.fir.cost_bits(),
                if in_force.iir.order > 0 {
                    filter.iir.cost_bits()
                } else {
                    0
                },
            );
        }
    }

    // With a second filter in front, the first is fitted to what it leaves.
    // The shift is shared, so it is capped where the second's taps still fit
    // the coefficient field.
    let mut filtered = [0.0f64; FIT_CONTEXT + crate::format::MAX_BLOCK];
    if let Some(second) = second.map(|index| SECOND_FILTERS[index]) {
        // The same design as the second filter in force is the same filter,
        // whatever shift its taps were quantised at: the residuals the decoder
        // holds are its history, and the shift is kept where it is so that
        // the taps need not be restated at all.
        let continuing = same_design(&in_force.iir, in_force.shift, second);
        let cap = if continuing {
            in_force.shift
        } else {
            shift_that_fits(second)
        };
        prefilter(&context[..total], second, &mut filtered[..total]);
        let fitted = lpc::fit(&filtered[..total], MAX_ORDER - second.len());
        for order in 1..=fitted.orders {
            let Some(quantised) = lpc::quantise_within(&fitted.coefficients[order - 1], order, cap)
            else {
                continue;
            };
            let mut taps = [0i32; MAX_ORDER];
            for (slot, tap) in taps.iter_mut().zip(second) {
                *slot = (tap * f64::from(1u32 << quantised.shift)).round() as i32;
            }
            let filter = Filter {
                fir: describe(&quantised.coefficients, order),
                iir: describe(&taps, second.len()),
                shift: quantised.shift,
            };
            // What would be written: the first filter, and the second only if
            // it is not already in force exactly as it is.
            let second_written = !(continuing
                && in_force.iir.same_taps(&filter.iir)
                && in_force.shift == filter.shift);
            let second = if second_written {
                filter.iir.cost_bits()
            } else {
                0
            };
            // From the history the decoder has — this pair's own residuals if
            // it is continuing, something else's if not. Even continuing, those
            // residuals were made with the *previous* first filter, and a
            // second filter's poles carry that difference across the block...
            if fill_residuals(&filter, &state, samples, scratch) <= MAX_HUFF_LSBS {
                consider(
                    &filter,
                    scratch,
                    &mut best,
                    &mut best_cost,
                    residuals,
                    filter.fir.cost_bits(),
                    second,
                );
            }
            // ...so also from the history this exact pair would have left,
            // installed along with it, which restates the second filter to
            // carry it. Whichever is cheaper.
            {
                let warm = history_after(&filter, &past[past.len().saturating_sub(WARM_UP)..]);
                let installed = install(&warm.residual, filter.iir.order);
                let mut start = state;
                start
                    .residual
                    .copy_from_slice(&installed.values[..MAX_IIR_ORDER]);
                let mut with_state = filter;
                with_state.iir.state = Some(installed);
                if fill_residuals(&with_state, &start, samples, scratch) <= MAX_HUFF_LSBS {
                    consider(
                        &with_state,
                        scratch,
                        &mut best,
                        &mut best_cost,
                        residuals,
                        with_state.fir.cost_bits(),
                        with_state.iir.cost_bits(),
                    );
                }
            }
        }
    }

    if best.filter.fir.order == 0 && best.filter.iir.order == 0 {
        fill_residuals(&best.filter, &state, samples, residuals);
    }
    (best, best_cost)
}

/// Whether taps carried at `shift` are this design, to within quantisation.
fn same_design(taps: &Taps, shift: u32, design: &[f64]) -> bool {
    taps.order == design.len()
        && taps.coefficients[..taps.order]
            .iter()
            .zip(design)
            .all(|(tap, wanted)| {
                (f64::from(*tap) / f64::from(1u32 << shift) - wanted).abs() < 0.004
            })
}

/// How many past samples a second filter is run over to find the history it
/// would have left. Its poles are inside 0.9, so sixty-four samples is well
/// past where the start stops mattering.
const WARM_UP: usize = 64;

/// The state a decoder would hold after running `filter` over `samples`
/// from rest.
fn history_after(filter: &Filter, samples: &[i32]) -> State {
    let mut state = State::default();
    for sample in samples {
        let prediction = filter.predict(&state);
        let residual = (i64::from(*sample) - prediction)
            .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
        state.signal.copy_within(0..MAX_ORDER - 1, 1);
        state.signal[0] = *sample;
        state.residual.copy_within(0..MAX_IIR_ORDER - 1, 1);
        state.residual[0] = residual;
    }
    state
}

/// A residual history as a state block can carry it: as many bits as the
/// widest value needs, up to the field's fifteen, and a shift for the rest.
/// The values kept are the ones the decoder will hold, low bits dropped by the
/// shift included, so the block is evaluated from exactly that.
fn install(residuals: &[i32; MAX_IIR_ORDER], order: usize) -> FilterState {
    let width = residuals[..order]
        .iter()
        .fold(1u32, |w, r| w.max(signed_width(*r)));
    let bits = width.min(MAX_STATE_BITS);
    let shift = width - bits;
    let mut state = FilterState {
        values: [0; MAX_ORDER],
        bits,
        shift,
    };
    for (slot, residual) in state.values.iter_mut().zip(&residuals[..order]) {
        *slot = (residual >> shift) << shift;
    }
    state
}

/// `x / (1 + B)`: what a second filter with taps `B` leaves the first one to
/// model, `y[t] = x[t] − Σ b_k · y[t − k]`, run from rest.
fn prefilter(samples: &[f64], taps: &[f64], out: &mut [f64]) {
    for t in 0..samples.len() {
        let mut y = samples[t];
        for (k, tap) in taps.iter().enumerate() {
            if t > k {
                y -= tap * out[t - k - 1];
            }
        }
        out[t] = y;
    }
}

/// The largest shift at which these taps still fit the coefficient field.
fn shift_that_fits(taps: &[f64]) -> u32 {
    let largest = taps.iter().fold(0.0f64, |m, t| m.max(t.abs()));
    let limit = f64::from(1i32 << (lpc::PRECISION - 1));
    let mut shift = lpc::MAX_SHIFT;
    while shift > 0 && largest * f64::from(1u32 << shift) >= limit {
        shift -= 1;
    }
    shift
}

/// Cost a candidate filter's residuals and keep it if it wins.
fn consider(
    filter: &Filter,
    candidate_residuals: &[i32],
    best: &mut Coding,
    best_cost: &mut usize,
    residuals: &mut [i32],
    first: usize,
    second: usize,
) {
    // A share of each description rather than the whole of it, because a
    // filter is described once and kept until it stops paying — see
    // [`crate::rate`], which holds the shares and what measured them.
    let description = crate::rate::filters(first, second);
    // Only a coding under what is left of the best cost once the description
    // is paid can win, so the search need not price anything at or above it.
    let budget = best_cost.saturating_sub(description);
    let Some((codebook, huff_lsbs, bits)) = best_entropy_within(candidate_residuals, budget) else {
        return;
    };
    let cost = bits + description;
    if cost < *best_cost {
        *best_cost = cost;
        *best = Coding {
            filter: *filter,
            codebook,
            huff_lsbs,
        };
        residuals.copy_from_slice(candidate_residuals);
    }
}

/// The cheapest way to write these residuals: which codebook, and how many
/// low bits under it.
///
/// The narrowest width that can hold the block is found from the block's
/// extremes and the codebook's stated span, in constant time per width, rather
/// than by trying widths against every sample. Then a few widths above that
/// are actually costed, because a wider field is not simply worse: it costs a
/// raw bit per sample and buys shorter codes, and where that trade turns over
/// depends on the shape of the block.
///
/// # The offset field is not searched, and that was measured
///
/// The format also carries a fifteen-bit offset that slides the codebook's
/// window along the number line, which would let a block the filter left
/// slightly biased fit a window one bit narrower. It was implemented and
/// measured: on a programme's centre channel it produced a **larger** stream
/// than leaving the offset at zero, for twenty per cent more encoding time.
/// That is what prediction residuals being close to zero-mean looks like from
/// the other side — the window is already where it should be, and changing it
/// costs fifteen bits that the narrower field does not repay over forty
/// samples.
///
/// [`crate::huffman`] still takes an offset, because it is part of the
/// format's definition of where a coding's range sits, and a later tool whose
/// residuals *are* biased may want it.
fn best_entropy(residuals: &[i32]) -> Option<(u8, u32, usize)> {
    best_entropy_within(residuals, usize::MAX)
}

/// As [`best_entropy`], among the codings that cost less than `budget`; `None`
/// if none does.
///
/// # It returns what the full search would, and why
///
/// The codings are tried in the same order, and one is kept only if it is
/// *strictly* cheaper than everything before it, so the first of equals wins
/// as before. What is new is only what is skipped: a block costs at least
/// `width + SHORTEST[codebook]` bits a sample, and that bound grows with the
/// width, so once it reaches the cheapest cost so far — or the budget — no
/// wider field in this codebook can be kept, and the sweep moves on to the
/// next. Nothing skipped could have been kept, so what is kept is the same.
fn best_entropy_within(residuals: &[i32], budget: usize) -> Option<(u8, u32, usize)> {
    /// How far above the narrowest usable width to look. Beyond this the
    /// symbols have all collapsed onto the middle of the table and the raw
    /// bits are pure loss.
    const SWEEP: u32 = 4;

    let lowest = i64::from(*residuals.iter().min()?);
    let highest = i64::from(*residuals.iter().max()?);

    let mut best: Option<(u8, u32, usize)> = None;
    // What a coding has to cost less than to be kept: the budget, then the
    // cheapest found.
    let mut bound = budget;
    for codebook in huffman::NONE..=huffman::MAX_CODEBOOK {
        // A coding's range only ever widens with the width, so the widths that
        // hold this block are everything from the narrowest one upwards, and
        // the narrowest is found by halving rather than by walking up from
        // zero and failing ten times.
        let holds = |width: u32| {
            let (low, high) = huffman::range(codebook, width, 0);
            lowest >= low && highest <= high
        };
        let estimate = estimated_width(codebook, lowest, highest);
        let Some(narrowest) = narrowest_width_from(holds, estimate) else {
            continue;
        };
        for width in narrowest..=(narrowest + SWEEP).min(MAX_HUFF_LSBS) {
            let floor = residuals.len() * (width as usize + huffman::SHORTEST[codebook as usize]);
            if floor >= bound {
                break;
            }
            // Every width from the narrowest up holds the block, so the
            // checks `huffman::cost` makes per sample would all pass.
            let bits = huffman::cost_fitting(codebook, width, residuals);
            if bits < bound {
                bound = bits;
                best = Some((codebook, width, bits));
            }
        }
    }
    best
}

/// The narrowest width `holds` accepts, or `None` if none up to the widest
/// does.
///
/// `holds` has to be monotone — false below the answer, true from it up —
/// which the format's codings are: a wider field only ever spans more.
///
/// Kept as what [`narrowest_width_from`] has to agree with.
#[cfg(test)]
fn narrowest_width(holds: impl Fn(u32) -> bool) -> Option<u32> {
    if !holds(MAX_HUFF_LSBS) {
        return None;
    }
    let mut low = 0u32;
    let mut high = MAX_HUFF_LSBS;
    while low < high {
        let middle = low + (high - low) / 2;
        if holds(middle) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    Some(low)
}

/// The narrowest width `holds` accepts, or `None` if none up to the widest
/// does, walking from `estimate` rather than halving from the whole range.
///
/// `holds` has to be monotone, as for [`narrowest_width`], and then this is
/// the same answer: if the estimate holds, the answer is the lowest width at
/// or under it that still does, and if it does not, the first width above it
/// that does. With an estimate within a width or two of the answer — see
/// [`estimated_width`] — that is two or three calls where the halving made six,
/// each one a pass over the codebook's range, twelve per candidate filter.
fn narrowest_width_from(holds: impl Fn(u32) -> bool, estimate: u32) -> Option<u32> {
    let mut width = estimate.min(MAX_HUFF_LSBS);
    if holds(width) {
        while width > 0 && holds(width - 1) {
            width -= 1;
        }
        return Some(width);
    }
    while width < MAX_HUFF_LSBS {
        width += 1;
        if holds(width) {
            return Some(width);
        }
    }
    None
}

/// Near the narrowest width a codebook holds a block in, from the block's
/// extremes: the codebooks span about `2^width` either side of nought with no
/// codebook, and about eight times that with one.
fn estimated_width(codebook: u8, lowest: i64, highest: i64) -> u32 {
    let reach = (-lowest).max(highest + 1).max(1) as u64;
    let bits = u64::BITS - (reach - 1).leading_zeros();
    if codebook == huffman::NONE {
        bits + 1
    } else {
        bits.saturating_sub(3)
    }
}

/// The filter state a decoder would hold after these: the last samples and the
/// last residuals, most recent first, zero-filled at the start of a stream
/// because that is what a decoder starts from.
pub fn state_from(past: &[i32], residuals: &[i32]) -> State {
    let mut state = State::default();
    for (slot, sample) in state.signal.iter_mut().zip(past.iter().rev()) {
        *slot = *sample;
    }
    for (slot, residual) in state.residual.iter_mut().zip(residuals.iter().rev()) {
        *slot = *residual;
    }
    state
}

/// Code a block with no prediction at all, and price it.
///
/// Which is what the first block after a restart header has to be. A filter
/// predicts from state a decoder joining the stream there does not have — and
/// the two decoders this project checks against disagree about what that state
/// is, one clearing it and one carrying it over, so a block that depends on it
/// is a block only one of them reconstructs. With no filter there is nothing
/// to depend on, and eight samples of it refill the state for the block after.
///
/// The entropy coding is still chosen: "no prediction" is not "no coding".
///
/// Returns [`usize::MAX`] for a block nothing can code, which cannot happen
/// for a 24-bit sample.
pub fn plain(samples: &[i32], residuals: &mut [i32]) -> (Coding, usize) {
    residuals.copy_from_slice(samples);
    match best_entropy(residuals) {
        Some((codebook, huff_lsbs, bits)) => (
            Coding {
                filter: Filter::default(),
                codebook,
                huff_lsbs,
            },
            bits,
        ),
        None => (
            Coding {
                filter: Filter::default(),
                codebook: huffman::NONE,
                huff_lsbs: MAX_HUFF_LSBS,
            },
            usize::MAX,
        ),
    }
}

/// Run the filter and return the width the residuals need.
///
/// The second filter feeds back: each residual becomes part of the state the
/// next prediction is made from. That loop can run away — a pole outside the
/// unit circle makes the residuals grow without bound — which is why the
/// magnitude is checked every sample rather than at the end. A candidate that
/// diverges is reported as unusably wide and dropped.
fn fill_residuals(filter: &Filter, start: &State, samples: &[i32], residuals: &mut [i32]) -> u32 {
    // Both histories as rings rather than arrays shifted along. The shift was
    // two memory moves a sample, in the innermost loop of the whole encoder;
    // both lengths are powers of two, so a ring costs an add and a mask. The
    // taps are read in the same order either way, so the sums are the same to
    // the bit.
    const SIGNAL_MASK: usize = MAX_ORDER - 1;
    const RESIDUAL_MASK: usize = MAX_IIR_ORDER - 1;
    const _: () = assert!(MAX_ORDER.is_power_of_two() && MAX_IIR_ORDER.is_power_of_two());

    let mut signal = start.signal;
    let mut past = start.residual;
    let mut newest = 0usize;
    let mut newest_residual = 0usize;
    let mut widest = 1;

    for (sample, residual) in samples.iter().zip(residuals.iter_mut()) {
        let mut accumulator = 0i64;
        for (back, coefficient) in filter.fir.coefficients[..filter.fir.order]
            .iter()
            .enumerate()
        {
            accumulator +=
                i64::from(*coefficient) * i64::from(signal[(newest + back) & SIGNAL_MASK]);
        }
        for (back, coefficient) in filter.iir.coefficients[..filter.iir.order]
            .iter()
            .enumerate()
        {
            accumulator +=
                i64::from(*coefficient) * i64::from(past[(newest_residual + back) & RESIDUAL_MASK]);
        }
        let prediction = accumulator >> filter.shift;

        let value = i64::from(*sample) - prediction;
        if value > i64::from(i32::MAX) || value < i64::from(i32::MIN) {
            return u32::MAX;
        }
        *residual = value as i32;
        widest = widest.max(signed_width(*residual));

        newest = (newest + SIGNAL_MASK) & SIGNAL_MASK;
        signal[newest] = *sample;
        newest_residual = (newest_residual + RESIDUAL_MASK) & RESIDUAL_MASK;
        past[newest_residual] = *residual;
    }
    widest
}

/// Turn quantised coefficients into the fields the stream carries.
///
/// The coefficients often share trailing zero bits — a consequence of the
/// shift the quantiser chose — and the format lets those be factored out into
/// `coeff_shift` so the field itself can be narrower.
fn describe(coefficients: &[i32; MAX_ORDER], order: usize) -> Taps {
    let mut widest = 1u32;
    let mut union = 0u32;
    for coefficient in &coefficients[..order] {
        widest = widest.max(magnitude_width(*coefficient));
        union |= coefficient.unsigned_abs();
    }
    let common = if union == 0 {
        0
    } else {
        union.trailing_zeros().min(7)
    };
    let coeff_bits = (widest - common).max(1);
    let coeff_shift = common.min(16 - coeff_bits);

    Taps {
        order,
        coefficients: *coefficients,
        coeff_bits,
        coeff_shift,
        state: None,
    }
}

/// Bits needed to hold `value` as a signed field, tightly.
#[inline]
fn signed_width(value: i32) -> u32 {
    let magnitude = if value < 0 { !value } else { value };
    33 - magnitude.leading_zeros()
}

/// Bits needed to hold `value` and its negation, which is what a coefficient
/// field has to do — the same value may appear with either sign in another
/// block and the width is chosen once.
#[inline]
fn magnitude_width(value: i32) -> u32 {
    if value == 0 {
        1
    } else {
        33 - value.unsigned_abs().leading_zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The search as it was before it was bounded, kept verbatim as what the
    /// bounded one has to agree with.
    fn best_entropy_unbounded(residuals: &[i32]) -> Option<(u8, u32, usize)> {
        const SWEEP: u32 = 4;
        let lowest = i64::from(*residuals.iter().min()?);
        let highest = i64::from(*residuals.iter().max()?);
        let mut best: Option<(u8, u32, usize)> = None;
        for codebook in huffman::NONE..=huffman::MAX_CODEBOOK {
            let holds = |width: u32| {
                let (low, high) = huffman::range(codebook, width, 0);
                lowest >= low && highest <= high
            };
            let Some(narrowest) = narrowest_width(holds) else {
                continue;
            };
            for width in narrowest..=(narrowest + SWEEP).min(MAX_HUFF_LSBS) {
                let Some(bits) = huffman::cost(codebook, width, 0, residuals) else {
                    continue;
                };
                if best.is_none_or(|(_, _, previous)| bits < previous) {
                    best = Some((codebook, width, bits));
                }
            }
        }
        best
    }

    /// Blocks of every shape the search meets: silence, a constant, small and
    /// large residuals, lopsided ones, the codec's extremes, one sample to a
    /// whole unit's worth.
    fn blocks() -> Vec<Vec<i32>> {
        let mut state = 0x9e37_79b9u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let mut out = vec![
            vec![0; 40],
            vec![7; 40],
            vec![-1],
            vec![(1 << 23) - 1, -(1 << 23)],
        ];
        for _ in 0..4000 {
            let len = 1 + next() as usize % 160;
            // How many bits a draw has, up to the codec's twenty-three.
            let scale = next() % 24;
            let bias = (next() % 5) as i32 - 2;
            let mut draw = || {
                if scale == 0 {
                    0
                } else {
                    (next() >> (32 - scale)) as i32
                }
            };
            out.push(
                (0..len)
                    .map(|_| {
                        // Two draws subtracted, so the block leans to the
                        // middle the way prediction residuals do, and some
                        // blocks lean off it.
                        let a = draw();
                        let b = draw();
                        (a - b + bias * (a >> 2)).clamp(-(1 << 23), (1 << 23) - 1)
                    })
                    .collect(),
            );
        }
        out
    }

    /// Walking from an estimate finds the width halving does, for every
    /// codebook and every pair of extremes — around nought, far from it, at
    /// the codec's ends and past what any width holds.
    #[test]
    fn the_narrowest_width_from_an_estimate_is_the_halved_one() {
        let mut state = 0x6a09_e667u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let mut pairs: Vec<(i64, i64)> = vec![
            (0, 0),
            (-1, -1),
            (0, 1),
            (-1, 0),
            (-7, 7),
            (-8, 7),
            (-7, 8),
            (-(1 << 23), (1 << 23) - 1),
            (-(1 << 30), 1 << 30),
            (5, 5),
            (-5, -5),
        ];
        for _ in 0..20_000 {
            let scale = next() % 32;
            let a = (i64::from(next() as i32)) >> (31 - scale.min(31));
            let b = (i64::from(next() as i32)) >> (31 - scale.min(31));
            pairs.push((a.min(b), a.max(b)));
        }
        for (lowest, highest) in pairs {
            for codebook in huffman::NONE..=huffman::MAX_CODEBOOK {
                let holds = |width: u32| {
                    let (low, high) = huffman::range(codebook, width, 0);
                    lowest >= low && highest <= high
                };
                assert_eq!(
                    narrowest_width_from(holds, estimated_width(codebook, lowest, highest)),
                    narrowest_width(holds),
                    "codebook {codebook}, [{lowest}, {highest}]"
                );
                // And from any estimate at all, however far off.
                for estimate in [0, 5, 12, 24, 40] {
                    assert_eq!(
                        narrowest_width_from(holds, estimate),
                        narrowest_width(holds),
                        "codebook {codebook}, [{lowest}, {highest}] from {estimate}"
                    );
                }
            }
        }
    }

    /// The unchecked cost is the checked one wherever the search calls it:
    /// every codebook, every width from the narrowest that holds the block.
    #[test]
    fn the_cost_of_a_fitting_coding_is_the_checked_cost() {
        for block in blocks() {
            let lowest = i64::from(*block.iter().min().unwrap());
            let highest = i64::from(*block.iter().max().unwrap());
            for codebook in huffman::NONE..=huffman::MAX_CODEBOOK {
                let holds = |width: u32| {
                    let (low, high) = huffman::range(codebook, width, 0);
                    lowest >= low && highest <= high
                };
                let Some(narrowest) = narrowest_width(holds) else {
                    continue;
                };
                for width in narrowest..=MAX_HUFF_LSBS {
                    assert_eq!(
                        Some(huffman::cost_fitting(codebook, width, &block)),
                        huffman::cost(codebook, width, 0, &block),
                        "codebook {codebook}, width {width}, {block:?}"
                    );
                }
            }
        }
    }

    /// The bounded search keeps exactly what the full one kept, ties included,
    /// and under a budget keeps it only when it is under the budget.
    #[test]
    fn the_bounded_search_keeps_what_the_full_one_kept() {
        for block in blocks() {
            let full = best_entropy_unbounded(&block);
            assert_eq!(best_entropy(&block), full, "{block:?}");
            let cost = full.map_or(0, |(_, _, bits)| bits);
            for budget in [0, cost.saturating_sub(1), cost, cost + 1, usize::MAX] {
                assert_eq!(
                    best_entropy_within(&block, budget),
                    full.filter(|(_, _, bits)| *bits < budget),
                    "budget {budget}, {block:?}"
                );
            }
        }
    }

    /// The first block of a restarting unit must not predict, because a
    /// decoder joining the stream there has nothing to predict from — and the
    /// two decoders this is checked against disagree about what it would have.
    #[test]
    fn an_unpredicted_block_is_its_own_samples() {
        let samples: Vec<i32> = (0i32..8).map(|n| (n - 4) * 12_345).collect();
        let mut residuals = vec![0i32; samples.len()];
        let (coding, bits) = plain(&samples, &mut residuals);

        assert_eq!(residuals, samples, "no prediction means no subtraction");
        assert_eq!(coding.filter.fir.order, 0);
        assert_eq!(coding.filter.iir.order, 0);
        assert!(bits > 0 && bits < samples.len() * 32);

        // And the state it leaves behind is determined by those samples alone,
        // which is the whole reason for the block.
        let state = state_from(&samples, &residuals);
        assert_eq!(state.signal[0], samples[7], "most recent first");
        assert_eq!(state.residual[0], samples[7]);
    }

    /// The whole point: what the decoder reconstructs is what went in.
    ///
    /// The decoder's arithmetic is written out here separately from the
    /// encoder's, so that a mistake shared between them would have to be made
    /// twice, in two directions — including which of the two histories
    /// receives the sample and which receives the residual, which is the one
    /// thing about the second filter that is easy to get backwards.
    fn reconstruct(coding: &Coding, start: &State, residuals: &[i32]) -> Vec<i32> {
        let mut state = *start;
        // A decoder installs the history a second filter arrives with.
        if let Some(installed) = &coding.filter.iir.state {
            state
                .residual
                .copy_from_slice(&installed.values[..MAX_IIR_ORDER]);
        }
        let mut out = Vec::with_capacity(residuals.len());
        for residual in residuals {
            let mut accumulator = 0i64;
            let fir = &coding.filter.fir;
            for (coefficient, past) in fir.coefficients[..fir.order].iter().zip(&state.signal) {
                accumulator += i64::from(*coefficient) * i64::from(*past);
            }
            let iir = &coding.filter.iir;
            for (coefficient, past) in iir.coefficients[..iir.order].iter().zip(&state.residual) {
                accumulator += i64::from(*coefficient) * i64::from(*past);
            }
            accumulator >>= coding.filter.shift;

            let sample = (accumulator + i64::from(*residual)) as i32;
            out.push(sample);
            state.signal.copy_within(0..MAX_ORDER - 1, 1);
            state.signal[0] = sample;
            state.residual.copy_within(0..MAX_IIR_ORDER - 1, 1);
            state.residual[0] = *residual;
        }
        out
    }

    fn tone(frames: usize, period: f64, amplitude: f64) -> Vec<i32> {
        (0..frames)
            .map(|n| ((n as f64 * std::f64::consts::TAU / period).sin() * amplitude) as i32)
            .collect()
    }

    #[test]
    fn residuals_reconstruct_the_samples_exactly() {
        for period in [3.0, 7.5, 40.0, 400.0] {
            for amplitude in [1.0, 1000.0, 8_000_000.0] {
                let samples = tone(40, period, amplitude);
                let past: Vec<i32> = Vec::new();
                let state = state_from(&past, &[]);
                let mut residuals = vec![0i32; samples.len()];
                let (coding, _) = choose(
                    &past,
                    state_from(&past, &[]),
                    &samples,
                    &mut residuals,
                    &Filter::default(),
                );
                let back = reconstruct(&coding, &state, &residuals);
                assert_eq!(
                    back, samples,
                    "period {period}, amplitude {amplitude}, orders {}/{}",
                    coding.filter.fir.order, coding.filter.iir.order
                );
            }
        }
    }

    /// Across blocks, with the history carried, which is where an off-by-one
    /// in the state would show up and nowhere else.
    #[test]
    fn the_history_carries_across_blocks() {
        let all = tone(400, 23.0, 4_000_000.0);
        let mut done = 0usize;
        let mut carried: Vec<i32> = Vec::new();
        while done < all.len() {
            let block = &all[done..(done + 40).min(all.len())];
            let state = state_from(&all[..done], &carried);
            let mut residuals = vec![0i32; block.len()];
            let (coding, _) = choose(
                &all[..done],
                state_from(&all[..done], &carried),
                block,
                &mut residuals,
                &Filter::default(),
            );
            assert_eq!(
                reconstruct(&coding, &state, &residuals),
                block,
                "block at {done}"
            );
            carried = residuals[residuals.len().saturating_sub(MAX_IIR_ORDER)..].to_vec();
            done += block.len();
        }
        let final_state = state_from(&all, &carried);
        assert_eq!(final_state.signal[0], all[all.len() - 1], "newest first");
        assert_eq!(final_state.signal[1], all[all.len() - 2]);
    }

    /// Nothing chooses a second filter, so nothing else exercises the half of
    /// the arithmetic that reads past residuals. This does, with one built by
    /// hand: encode a block through it and reconstruct the samples back.
    ///
    /// Worth keeping precisely because the search does not go here. A stream
    /// may carry such a filter, the writer can express one, and an encoder
    /// whose model of the format quietly rots in the corner nobody visits is
    /// how a later change breaks in a way that takes a day to find.
    #[test]
    fn a_second_filter_round_trips_too() {
        let samples = tone(60, 11.0, 2_000_000.0);
        let filter = Filter {
            fir: Taps {
                order: 2,
                coefficients: [7000, -4000, 0, 0, 0, 0, 0, 0],
                coeff_bits: 15,
                coeff_shift: 0,
                state: None,
            },
            iir: Taps {
                order: 3,
                coefficients: [2000, -900, 400, 0, 0, 0, 0, 0],
                coeff_bits: 15,
                coeff_shift: 0,
                state: None,
            },
            shift: 12,
        };
        let state = State {
            signal: [11, -22, 33, -44, 55, -66, 77, -88],
            residual: [5, -6, 7, -8],
        };

        let mut residuals = vec![0i32; samples.len()];
        let width = fill_residuals(&filter, &state, &samples, &mut residuals);
        assert!(width < 32, "the filter diverged");
        assert!(
            residuals.iter().any(|r| *r != 0),
            "a filter that predicts perfectly is not a test"
        );

        let coding = Coding {
            filter,
            codebook: huffman::NONE,
            huff_lsbs: width,
        };
        assert_eq!(reconstruct(&coding, &state, &residuals), samples);
    }

    /// A predictable signal has to cost fewer bits than its samples do, or the
    /// search is not doing anything.
    #[test]
    fn a_predictable_signal_gets_a_filter() {
        let samples = tone(40, 40.0, 4_000_000.0);
        let mut residuals = vec![0i32; samples.len()];
        let (coding, _) = choose(
            &[],
            State::default(),
            &samples,
            &mut residuals,
            &Filter::default(),
        );
        assert!(
            coding.filter.fir.order > 0,
            "a smooth tone should be predicted"
        );
        assert!(
            coding.huff_lsbs < 23,
            "a smooth tone should not need {} bits a sample",
            coding.huff_lsbs
        );
    }

    /// And an unpredictable one must not be made worse by trying.
    #[test]
    fn noise_falls_back_rather_than_growing() {
        let mut state = 0x1234_5678u32;
        let samples: Vec<i32> = (0..40)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as i32 & 0x00ff_ffff) - 0x0080_0000
            })
            .collect();
        let mut residuals = vec![0i32; samples.len()];
        let (coding, _) = choose(
            &[],
            State::default(),
            &samples,
            &mut residuals,
            &Filter::default(),
        );

        // What sending it raw would have cost, which is the fallback the
        // search always has available.
        let widest = samples.iter().fold(1, |m, s| m.max(signed_width(*s))) as usize;
        // Sending it raw still costs the two order fields that say there is no
        // filter, so that is what the comparison has to include — not a
        // tolerance chosen to make it pass.
        let raw = widest * samples.len() + Filter::default().cost_bits();
        // A filter is charged a quarter of its description, because one is
        // kept until it stops paying — so that is the accounting the search
        // is held to here, not the whole description against one block.
        let chosen = huffman::cost(coding.codebook, coding.huff_lsbs, 0, &residuals)
            .expect("the chosen coding fits its own residuals")
            + crate::rate::filters(coding.filter.cost_bits(), 0);
        assert!(
            chosen <= raw,
            "coding noise cost {chosen} bits against {raw} for sending it raw"
        );
    }

    /// The closed form and the writer have to agree, or a candidate is priced
    /// at one width and written at another — which shows up as an encoder that
    /// prefers filters it should not, and never as a failure.
    #[test]
    fn the_closed_form_agrees_with_the_counter() {
        use hz_core::bits::BitCounter;

        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for order in 0..=MAX_ORDER {
            for _ in 0..40 {
                let coeff_bits = 1 + (next() % 16) as u32;
                let coeff_shift = (next() % (17 - coeff_bits) as u64) as u32;
                let mut coefficients = [0i32; MAX_ORDER];
                for slot in coefficients[..order].iter_mut() {
                    let limit = 1i64 << (coeff_bits - 1);
                    let stored = (next() % (2 * limit) as u64) as i64 - limit;
                    *slot = (stored as i32) << coeff_shift;
                }
                // Half of them carry a history, which only a second filter may
                // and which is the part of the description easiest to forget.
                let state_block = (next() % 2 == 0 && order > 0).then(|| {
                    let bits = 1 + (next() % 15) as u32;
                    let shift = (next() % 16) as u32;
                    let mut values = [0i32; MAX_ORDER];
                    for slot in values[..order].iter_mut() {
                        let limit = 1i64 << (bits - 1);
                        let stored = (next() % (2 * limit) as u64) as i64 - limit;
                        *slot = (stored as i32) << shift;
                    }
                    FilterState {
                        values,
                        bits,
                        shift,
                    }
                });
                let taps = Taps {
                    order,
                    coefficients,
                    coeff_bits,
                    coeff_shift,
                    state: state_block,
                };

                let mut counter = BitCounter::new();
                crate::frame::write_filter_params(&mut counter, &taps, 9);
                assert_eq!(
                    taps.cost_bits(),
                    counter.bits(),
                    "order {order}, {coeff_bits} bits shifted {coeff_shift}, history {}",
                    taps.state.is_some()
                );
            }
        }
    }

    #[test]
    fn silence_is_not_a_filter() {
        let samples = vec![0i32; 40];
        let mut residuals = vec![0i32; 40];
        let (coding, _) = choose(
            &[],
            State::default(),
            &samples,
            &mut residuals,
            &Filter::default(),
        );
        assert_eq!(coding.filter.fir.order, 0);
        assert_eq!(coding.filter.iir.order, 0);
        assert_eq!(coding.codebook, huffman::NONE);
        assert_eq!(
            coding.huff_lsbs, 0,
            "a silent channel should cost no bits at all"
        );
    }

    /// The coefficient fields have hard limits, and a stream that breaks them
    /// is rejected outright rather than decoded wrongly.
    #[test]
    fn coefficient_fields_stay_inside_what_the_format_allows() {
        let mut state = 0x9e37_79b9u32;
        for _ in 0..200 {
            let samples: Vec<i32> = (0..40)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((state >> 9) as i32 & 0x000f_ffff) - 0x0008_0000
                })
                .collect();
            let mut residuals = vec![0i32; 40];
            let (coding, _) = choose(
                &[],
                State::default(),
                &samples,
                &mut residuals,
                &Filter::default(),
            );
            let filter = coding.filter;
            assert!(filter.fir.order <= MAX_ORDER);
            assert!(filter.iir.order <= MAX_IIR_ORDER);
            assert!(coding.huff_lsbs < 32, "huff_lsbs is five bits");
            if filter.fir.order > 0 || filter.iir.order > 0 {
                assert!(
                    (crate::lpc::MIN_SHIFT..16).contains(&filter.shift),
                    "shift {} is outside what decoders accept",
                    filter.shift
                );
            }
            for taps in [&filter.fir, &filter.iir] {
                if taps.order == 0 {
                    continue;
                }
                assert!((1..=16).contains(&taps.coeff_bits));
                assert!(
                    taps.coeff_bits + taps.coeff_shift <= 16,
                    "{} + {} exceeds the format's limit",
                    taps.coeff_bits,
                    taps.coeff_shift
                );
                assert!(taps.coeff_shift <= 7, "coeff_shift is three bits");
                for coefficient in &taps.coefficients[..taps.order] {
                    let stored = coefficient >> taps.coeff_shift;
                    assert_eq!(
                        stored << taps.coeff_shift,
                        *coefficient,
                        "a coefficient did not survive being factored"
                    );
                    let limit = 1i32 << (taps.coeff_bits - 1);
                    assert!(
                        (-limit..limit).contains(&stored),
                        "{stored} does not fit {} signed bits",
                        taps.coeff_bits
                    );
                    assert_ne!(*coefficient, -32768, "a value decoders reject");
                }
            }
        }
    }
}
