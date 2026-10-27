"""Alternative readouts: posterior -> events.

EXP-12 measured +0.6365 of combined score sitting in the readout alone, on
posteriors we already have. EXP-11/12 also diagnosed exactly why the naive
threshold loses it:

  * 76% of missed events are SEEN (posterior crosses threshold inside them)
  * 88% of predictions land on a reference
  * for cleanly 1:1 matched events, boundary error is ~zero bias but 0.25 s
    STANDARD DEVIATION - while a 0.4 s event only tolerates +/-0.08 s
  * 23.7% of reference events are swallowed by a merged span

The root cause is that a sigmoid posterior RAMPS. Thresholding a smooth slope
turns gradual confidence into an arbitrary edge, and median filtering makes it
worse. These methods estimate edges instead of slicing them.
"""
from __future__ import annotations

import numpy as np


def _runs(mask):
    if not mask.any():
        return []
    d = np.diff(mask.astype(np.int8), prepend=0, append=0)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def hysteresis(p, hi=0.6, lo=0.3, frame_sec=0.02, min_dur=0.05, merge_gap=0.0):
    """Two-threshold (Schmitt) detection.

    A region must exceed `hi` somewhere to be accepted, but its extent is taken
    out to where the posterior falls below `lo`. This is the standard fix for
    ramping signals: the high threshold controls WHETHER, the low threshold
    controls WHERE. A single threshold has to do both and does neither well.
    """
    p = np.asarray(p, np.float32).ravel()
    strong, weak = p >= hi, p >= lo
    out = []
    for a, b in _runs(weak):
        if strong[a:b].any():
            out.append([a * frame_sec, b * frame_sec])
    if merge_gap > 0 and out:
        m = [out[0]]
        for s, e in out[1:]:
            if s - m[-1][1] <= merge_gap:
                m[-1][1] = e
            else:
                m.append([s, e])
        out = m
    return [(s, e) for s, e in out if e - s >= min_dur]


def peak_split(p, thr=0.5, frame_sec=0.02, min_dur=0.05, prominence=0.15,
               min_sep=0.10):
    """Threshold, then SPLIT each run at internal minima.

    Directly attacks the merge problem (23.7% of reference events are swallowed
    by a merged span). Within a detected region, a dip of at least `prominence`
    below the surrounding peaks marks a boundary between two events.
    """
    p = np.asarray(p, np.float32).ravel()
    out = []
    sep = max(1, int(min_sep / frame_sec))
    for a, b in _runs(p >= thr):
        seg = p[a:b]
        if len(seg) < 3:
            out.append((a * frame_sec, b * frame_sec))
            continue
        cuts = []
        for i in range(1, len(seg) - 1):
            if seg[i] <= seg[i - 1] and seg[i] <= seg[i + 1]:
                left = seg[:i].max() if i else seg[0]
                right = seg[i + 1:].max() if i + 1 < len(seg) else seg[-1]
                if min(left, right) - seg[i] >= prominence:
                    if not cuts or i - cuts[-1] >= sep:
                        cuts.append(i)
        bounds = [0] + cuts + [len(seg)]
        for s, e in zip(bounds[:-1], bounds[1:]):
            if e > s:
                out.append(((a + s) * frame_sec, (a + e) * frame_sec))
    return [(s, e) for s, e in out if e - s >= min_dur]


def derivative_edges(p, thr=0.5, frame_sec=0.02, min_dur=0.05, smooth=2):
    """Place edges at the steepest rise/fall rather than at a level crossing.

    For a ramping posterior the level crossing depends on the threshold, but the
    inflection does not. Detect regions by threshold, then move each boundary to
    the local extremum of the derivative within a short search window.
    """
    p = np.asarray(p, np.float32).ravel()
    if smooth > 0:
        k = np.ones(2 * smooth + 1, np.float32) / (2 * smooth + 1)
        ps = np.convolve(p, k, mode="same")
    else:
        ps = p
    d = np.gradient(ps)
    W = max(2, int(0.12 / frame_sec))
    out = []
    for a, b in _runs(p >= thr):
        lo = max(0, a - W)
        on = lo + int(np.argmax(d[lo:min(a + W, len(d))])) if min(a + W, len(d)) > lo else a
        hi = min(len(d), b + W)
        off = max(0, b - W) + int(np.argmin(d[max(0, b - W):hi])) if hi > max(0, b - W) else b
        off = max(off, on + 1)
        out.append((on * frame_sec, off * frame_sec))
    return [(s, e) for s, e in out if e - s >= min_dur]


def adaptive_threshold(p, rel=0.5, floor=0.25, frame_sec=0.02, min_dur=0.05,
                       merge_gap=0.0):
    """Per-clip threshold set relative to that clip's own posterior range.

    Clips differ in overall confidence (recording level, language, speaker).
    A global threshold is too high for quiet clips and too low for loud ones;
    EXP-10 measured +0.218 combined from a per-clip ORACLE threshold, so a
    principled per-clip rule should recover part of that.
    """
    p = np.asarray(p, np.float32).ravel()
    t = max(floor, float(p.min() + rel * (p.max() - p.min())))
    out = [(a * frame_sec, b * frame_sec) for a, b in _runs(p >= t)]
    if merge_gap > 0 and out:
        m = [list(out[0])]
        for s, e in out[1:]:
            if s - m[-1][1] <= merge_gap:
                m[-1][1] = e
            else:
                m.append([s, e])
        out = [tuple(x) for x in m]
    return [(s, e) for s, e in out if e - s >= min_dur]


def hysteresis_split(p, hi=0.6, lo=0.3, frame_sec=0.02, min_dur=0.05,
                     prominence=0.15, min_sep=0.10):
    """Hysteresis for extent, then peak-splitting for merges. Both fixes at once."""
    p = np.asarray(p, np.float32).ravel()
    strong, weak = p >= hi, p >= lo
    out = []
    sep = max(1, int(min_sep / frame_sec))
    for a, b in _runs(weak):
        if not strong[a:b].any():
            continue
        seg = p[a:b]
        cuts = []
        for i in range(1, len(seg) - 1):
            if seg[i] <= seg[i - 1] and seg[i] <= seg[i + 1]:
                left = seg[:i].max() if i else seg[0]
                right = seg[i + 1:].max() if i + 1 < len(seg) else seg[-1]
                if min(left, right) - seg[i] >= prominence:
                    if not cuts or i - cuts[-1] >= sep:
                        cuts.append(i)
        bounds = [0] + cuts + [len(seg)]
        for s, e in zip(bounds[:-1], bounds[1:]):
            if e > s and p[a + s:a + e].max() >= hi:
                out.append(((a + s) * frame_sec, (a + e) * frame_sec))
    return [(s, e) for s, e in out if e - s >= min_dur]


def refine(spans, dist, frame_sec=0.02, radius=0.20, min_votes=3,
           max_move=0.35):
    """Correct each span's boundaries using the boundary-regression head.

    Args:
        spans:  [(onset, offset)] from any threshold readout.
        dist:   (T, 2) predicted signed seconds to the nearest onset / offset.
                Positive means the edge lies BEHIND this frame.
        radius: every frame within +-radius of a provisional boundary votes.
        max_move: refuse a correction larger than this; a wild vote should not
                be able to destroy a boundary that the threshold got right.

    Each voting frame t proposes `t - dist[t]` as the edge location, and we take
    the MEDIAN of the proposals. Three properties matter and none of them are
    available to a threshold crossing:

      * many frames vote on one boundary, so independent errors average down;
      * the estimate is continuous in time - not quantised to the 20 ms grid -
        which matters when the tolerance is only 4.3 frames wide;
      * it is independent of the posterior's amplitude, so a slow ramp no
        longer drags the boundary with it.

    The median rather than the mean is deliberate: frames that happen to sit
    nearer a DIFFERENT edge vote for that one, and those votes are outliers, not
    noise. A mean would follow them.
    """
    T = len(dist)
    R = max(1, int(round(radius / frame_sec)))
    out = []
    for on, off in spans:
        new = []
        for k, t0 in ((0, on), (1, off)):
            i = int(round(t0 / frame_sec))
            a, b = max(0, i - R), min(T, i + R + 1)
            if b - a < min_votes:
                new.append(t0)
                continue
            idx = np.arange(a, b)
            votes = (idx + 0.5) * frame_sec - dist[a:b, k]
            t1 = float(np.median(votes))
            new.append(t1 if abs(t1 - t0) <= max_move else t0)
        if new[1] > new[0]:
            out.append((max(0.0, new[0]), new[1]))
    return sorted(out)


def bias_correct(spans, d_on, d_off):
    """Shift every boundary by a constant. Degenerate spans are dropped."""
    out = []
    for a, b in spans:
        a2, b2 = max(0.0, a + d_on), b + d_off
        if b2 > a2:
            out.append((a2, b2))
    return sorted(out)


def routed_correct(spans, p_syn, nat=(0.0, 0.0), syn=(0.03, -0.04)):
    """Bias-correct with the shift BLENDED by P(this clip is synthetic).

    EXP-27 measured opposite situations in the two populations: synthetic spans
    come out ~0.075 s too narrow (onset +0.0395 late, offset -0.0355 early) on
    top of a tight std of 0.069, while natural spans have ~zero bias and a std
    of 0.275. Correcting the synthetic bias is worth +0.0906 combined there and
    nothing on natural, so one global shift throws most of it away.

    Blending rather than hard-routing is deliberate. The classifier is 97.3%
    accurate, not perfect, and a wrong hard decision applies a 0.075 s shift to
    a clip that wanted none. Weighting by probability makes the cost of an
    uncertain clip proportional to the uncertainty.
    """
    d_on = (1 - p_syn) * nat[0] + p_syn * syn[0]
    d_off = (1 - p_syn) * nat[1] + p_syn * syn[1]
    return bias_correct(spans, d_on, d_off)


def tolerance_window(spans, p, frame_sec=0.02, tol_frac=0.20, tol_floor=0.05,
                     search=0.30, iters=2):
    """Bayes-optimal boundary placement under a tolerance metric.

    Derivation. The frame posterior is p(t) = P(event is active at t). For a
    rising edge that is exactly the CDF of the ONSET's position: p(t) = P(onset
    <= t). So the probability density of the onset being at t is dp/dt, and the
    probability that a guess t_hat lands within tolerance is

        P(|t_hat - onset| <= tol) = integral of dp/dt over [t_hat-tol, t_hat+tol]
                                  = p(t_hat + tol) - p(t_hat - tol)

    The scorer's loss is exactly 0/1 at that tolerance, so the optimal estimate
    maximises that quantity:

        onset*  = argmax_t  [ p(t+tol) - p(t-tol) ]
        offset* = argmax_t  [ p(t-tol) - p(t+tol) ]        (falling edge)

    A threshold crossing answers a different question - "where does p pass
    0.5?" - whose answer moves with the ramp's HEIGHT as well as its position.
    A quiet event whose posterior peaks at 0.6 crosses 0.5 late on the rise and
    early on the fall, so its span comes out too narrow; a confident event
    crosses early and late and comes out too wide. That coupling is pure error,
    and it is what EXP-27 measured as a 0.075 s systematic narrowing on
    synthetic clips.

    This estimator is indifferent to the ramp's height. It only asks where the
    probability MASS of the transition sits, which is the question the metric
    asks. When the ramp is much narrower than the tolerance the two agree; when
    it is wider - natural clips, 0.200 s against 0.086 s - they diverge, and
    this one is right.

    `tol` depends on the reference duration, which we do not have, so we use
    the predicted duration and iterate: refine boundaries, recompute tol from
    the new duration, refine again.
    """
    T = len(p)
    out = []
    for a, b in spans:
        for _ in range(max(1, iters)):
            tol = max(tol_frac * (b - a), tol_floor)
            k = max(1, int(round(tol / frame_sec)))
            r = max(1, int(round(search / frame_sec)))

            def best(centre, rising):
                """Centroid of the transition density, not its argmax.

                MEASURED CORRECTION. The plain argmax of p(t+tol) - p(t-tol) is
                DEGENERATE whenever the ramp is narrower than the tolerance:
                every window that contains the whole transition scores the
                same, so argmax returns the first index of a flat plateau and
                throws away a boundary that was already precise. On synthetic
                clips (0.040 s ramp against an 0.086 s tolerance) that cost
                -0.20 combined.

                The fix keeps the derivation and fixes the tie-break. The
                transition density is dp/dt, and its CENTROID is the unique
                point that sits in the middle of every optimal window: for a
                narrow spike it returns the spike, for a wide ramp it returns
                the centre of mass. No degeneracy either way.
                """
                i0 = int(round(centre / frame_sec))
                lo, hi = max(1, i0 - r), min(T - 1, i0 + r)
                if hi <= lo + 1:
                    return centre
                d = np.diff(p[lo - 1:hi + 1])
                w = np.clip(d if rising else -d, 0.0, None)
                if w.sum() <= 1e-6:
                    return centre
                idx = np.arange(lo, lo + len(w))
                return float((idx * w).sum() / w.sum() * frame_sec)

            a2, b2 = best(a, True), best(b, False)
            if b2 - a2 < frame_sec:
                break
            a, b = a2, b2
        if b > a:
            out.append((max(0.0, a), b))
    return sorted(out)


def asymmetric(p, thr_on=0.6, thr_off=0.7, frame_sec=0.02, min_dur=0.10,
               merge_gap=0.06):
    """One span, two thresholds: one for the rising edge, one for the falling.

    EXP-28 measured why this is not just another knob. On natural clips the
    posterior's FALL is 36% slower than its RISE (median 0.300 s vs 0.220 s),
    because sounds attack quickly and decay slowly. A single threshold has to
    serve both edges, and whichever value suits the sharp attack is wrong for
    the long decay. Splitting them lets each edge be read at the level where
    its own transition is steepest.

    This is the eleventh readout variant we have tried and the first to help
    (+0.0158 combined on natural clips) - because it follows a measured
    asymmetry rather than searching a grid.
    """
    core = p >= max(thr_on, thr_off)
    if not core.any():
        return []
    loose = p >= min(thr_on, thr_off)
    out = []
    for s, e in _runs(loose):
        if not core[s:e].any():
            continue
        a = s
        while a > 0 and p[a - 1] >= thr_on:
            a -= 1
        b = e
        while b < len(p) and p[b - 1] >= thr_off:
            b += 1
        out.append((a * frame_sec, min(len(p), b) * frame_sec))
    merged = []
    for a, b in sorted(out):
        if merged and a - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [(a, b) for a, b in merged if b - a >= min_dur]


def split_on_crossings(spans, dist, frame_sec=0.02, min_dur=0.08,
                       min_gap=0.10):
    """Split a merged span using the boundary head's ZERO CROSSINGS.

    EXP-19 attributed 22.6% of undetected events to MERGED spans - two
    reference events swallowed by one prediction - and EXP-20 showed the
    posterior has no visible valley between them, so `peak_split` made things
    worse at every prominence (0.277 -> 0.19).

    The distance field answers a different question and does have the
    information. `dist[t, 0]` is the signed seconds from frame t to the nearest
    ONSET, so it crosses zero, rising, exactly at each onset; `dist[t, 1]` does
    the same at each offset. A span containing two rising zero crossings of the
    onset channel contains two events, whether or not the posterior dipped
    between them.

    This is the same mechanism as `refine`, used for counting rather than
    placing. It costs nothing at inference and is only accepted if it wins its
    sweep, like everything else.
    """
    T = len(dist)
    out = []
    for a, b in spans:
        i0, i1 = max(0, int(a / frame_sec)), min(T - 1, int(b / frame_sec))
        if i1 - i0 < 3:
            out.append((a, b))
            continue
        d_on = dist[i0:i1 + 1, 0]
        d_off = dist[i0:i1 + 1, 1]
        # rising zero crossings of the onset channel, strictly inside the span
        cross = [i0 + k + 1 for k in range(len(d_on) - 1)
                 if d_on[k] < 0 <= d_on[k + 1]]
        cross = [c for c in cross
                 if (c - i0) * frame_sec >= min_gap
                 and (i1 - c) * frame_sec >= min_gap]
        if not cross:
            out.append((a, b))
            continue
        # cut just before each extra onset, at the matching offset crossing if
        # there is one nearby, otherwise at the onset itself
        cuts = []
        for c in cross:
            k = c - i0
            lo = max(0, k - int(0.15 / frame_sec))
            seg = d_off[lo:k]
            j = None
            for m in range(len(seg) - 1):
                if seg[m] < 0 <= seg[m + 1]:
                    j = lo + m + 1
            cuts.append((i0 + j) if j is not None else c)
        edges = [i0] + sorted(set(cuts)) + [i1]
        pieces = [(edges[k] * frame_sec, edges[k + 1] * frame_sec)
                  for k in range(len(edges) - 1)]
        pieces = [(x, y) for x, y in pieces if y - x >= min_dur]
        out.extend(pieces if pieces else [(a, b)])
    return sorted(out)


def _rising_zeros(d, frame_sec=0.02):
    """Sub-frame times where a signed-distance channel crosses zero upward.

    dist[t] = t - (nearest edge), so it is a sawtooth: it rises through zero at
    every edge, then jumps back down when a later edge becomes the nearest one.
    The rising crossings ARE the edges. Linear interpolation between the two
    straddling frames gives sub-frame resolution, which matters when the whole
    tolerance is 4.3 frames wide.
    """
    out = []
    for k in range(len(d) - 1):
        a, b = float(d[k]), float(d[k + 1])
        if a < 0.0 <= b and (b - a) > 0:
            out.append((k + a / (a - b) if (a - b) != 0 else k) * frame_sec)
    return out


def propose_from_boundaries(dist, p, frame_sec=0.02, min_conf=0.35,
                            min_dur=0.06, max_dur=12.0):
    """Detect events from the boundary field alone, not from a threshold.

    A different detector, available only once the distance head is accurate.
    `refine` uses the head to MOVE boundaries that thresholding already found;
    this uses it to FIND them. Onsets are the rising zero crossings of channel
    0, offsets those of channel 1, each onset is paired with the first offset
    after it, and a span survives only if the posterior inside it averages at
    least `min_conf` - the posterior is demoted from locating events to merely
    confirming them.

    Worth trying because the two mechanisms fail differently. Thresholding
    misses an event whose posterior never clears the bar (EXP-19: 26.8% of
    misses are 'suppressed', a further 10.1% 'blind'), while the distance head
    can still point at an edge it is only moderately sure about. Whether that
    is a net gain is a question for the sweep, not for me.
    """
    ons = _rising_zeros(dist[:, 0], frame_sec)
    offs = _rising_zeros(dist[:, 1], frame_sec)
    if not ons or not offs:
        return []
    T = len(p)
    out = []
    j = 0
    for a in ons:
        while j < len(offs) and offs[j] <= a:
            j += 1
        if j >= len(offs):
            break
        b = offs[j]
        if not (min_dur <= b - a <= max_dur):
            continue
        ia, ib = max(0, int(a / frame_sec)), min(T, max(1, int(b / frame_sec)))
        if ib > ia and float(p[ia:ib].mean()) >= min_conf:
            out.append((a, b))
    # a later onset may pair to the same offset; keep the tightest span
    out.sort()
    merged = []
    for a, b in out:
        if merged and abs(merged[-1][1] - b) < 1e-6:
            merged[-1] = (max(merged[-1][0], a), b)
        else:
            merged.append((a, b))
    return sorted(merged)


def union_events(a, b, iou_thresh=0.5):
    """Add spans from `b` that do not already overlap something in `a`.

    Used to let the boundary-field detector contribute events the threshold
    missed without disturbing the ones it found. Overlap is IoU so a near
    duplicate is dropped rather than emitted twice - Event-F1 matches
    one-to-one, so a duplicate is a guaranteed false positive.
    """
    out = list(a)
    for s, e in b:
        keep = True
        for s2, e2 in a:
            inter = max(0.0, min(e, e2) - max(s, s2))
            union = max(e, e2) - min(s, s2)
            if union > 0 and inter / union >= iou_thresh:
                keep = False
                break
        if keep:
            out.append((s, e))
    return sorted(out)
