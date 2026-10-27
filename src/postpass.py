"""Decode chain of the final submission (LB 1.5926), as one tested module.

Order (per clip, routed by submit_ens.natural_by_peak):
  champion decode  natural  : mean(nat_members) -> readout md02_natural (perpop hysteresis + ch7 overlay)
                   synthetic: mean(syn_members) -> readout ens5_v5w2_ov -> drop events < level_db vs the clip
  natural  : bgext (extend long background events while p0 >= low) -> habit full-span event -> type-C inner events
  synthetic: two-step edge shift (+4/-4 ms clamped, then +4/-2 ms)
Ported verbatim from the scripts that built the uploaded files (build_bg.py, build_syn.py, build_habit.py,
build_typec.py, build_syn86.py, build_chain.py); configs/postpass.json holds every constant."""
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

import ens_gate as E
import postproc
import submit_ens as SE
import submit_ft as SF

REPO = Path(__file__).resolve().parent.parent


def load_cfg(path=None):
    return json.load(open(path or REPO / "configs" / "postpass.json"))


def load_readouts(cfg):
    nat = json.load(open(REPO / cfg["readouts"]["natural"]))
    syn = json.load(open(REPO / cfg["readouts"]["synthetic"]))["readout"]
    return nat, syn


def r3(evs):
    return [tuple(round(x, 3) for x in e[:2]) for e in evs if e[1] > e[0]]


def decode(members, ro, ids, nat, dur, C, q=0.7):
    """Champion decode on the mean posterior of `members` (a list; repeats weight a member)."""
    post8 = E.average([C[t]["post"] for t in members])
    dist = E.average([C[t]["dist"] for t in members])
    post = {c: post8[c][:, 0] for c in ids}
    p = SE.decode_routed(post, dist, ids, ro, nat)
    p = SE.apply_overlay(p, post8, ids, ro.get("overlay"), nat)
    return postproc.clip_to_duration(postproc.empty_clip_fallback(p, post, q=q), dur)


def level_filter(pr, ids, nat, arr, clips, level_db=-16.0):
    """Synthetic clips only: drop events quieter than level_db relative to the whole clip."""
    out = dict(pr)
    for c in ids:
        if nat[c] or not pr[c]:
            continue
        e = clips[c]
        y = np.asarray(arr[e["off"]:e["off"] + e["n"]], dtype=np.float32) / 32768.0
        cdb = 10 * np.log10(np.mean(y ** 2) + 1e-12)
        keep = []
        for ev in pr[c]:
            a = int(ev[0] * 16000)
            b = max(a + 1, int(ev[1] * 16000))
            if 10 * np.log10(np.mean(y[a:b] ** 2) + 1e-12) - cdb >= level_db:
                keep.append(ev)
        out[c] = keep
    return out


def bgext(evs, p8, dur, cfg, fs=0.02):
    """Extend long background-category events while p0 >= low; bridge short gaps between long events."""
    cats, low, bridge, minlen = set(cfg["cats"]), cfg["low"], cfg["bridge"], cfg["minlen"]
    p0 = p8[:, 0]; n = len(p0); out = []
    for on, off in evs:
        a, b = int(round(on / fs)), min(n, int(round(off / fs)))
        if b > a and off - on >= minlen and int(np.argmax(p8[a:b, 1:].mean(0))) in cats:
            while a > 0 and p0[a - 1] >= low: a -= 1
            while b < n and p0[b] >= low: b += 1
            out.append((round(a * fs, 3), round(min(b * fs, dur), 3)))
        else:
            out.append((on, off))
    out.sort(); m = []
    for e in out:
        if m and e[0] - m[-1][1] <= bridge and m[-1][1] - m[-1][0] >= minlen and e[1] - e[0] >= minlen:
            g0, g1 = int(m[-1][1] / fs), int(e[0] / fs)
            if g1 <= g0 or p0[g0:g1].min() >= low:
                m[-1] = (m[-1][0], max(m[-1][1], e[1])); continue
        m.append(e)
    return r3(m)


def habit_span(evs, p, dur, cfg):
    """Add a full-clip event when the habit model says annotators label long background (p >= thr) and no
    existing event already covers >= max_cover of the clip."""
    if p is not None and p >= cfg["thr"] and max([(b - a) / dur for a, b in evs] or [0]) < cfg["max_cover"]:
        return sorted(evs + [(0.0, round(dur, 3))])
    return evs


def _runs(m, gap, fs):
    out, i, n = [], 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]: j += 1
            if out and (i - out[-1][1]) * fs <= gap: out[-1] = (out[-1][0], j)
            else: out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _dup(e, P):
    tol = max(.2 * (e[1] - e[0]), .05)
    return any(abs(e[0] - p[0]) <= tol and abs(e[1] - p[1]) <= tol for p in P)


def typec_events(evs, p8, dur, is_habit_clip, cfg, fs=0.02):
    """Type-C inner events: a category run (score >= theta, >= minlen) nested inside a longer live event
    (>= hmult x its length) whose dominant category differs. The habit full-span event is never a host."""
    th, minlen, hmult, gap = cfg["theta"], cfg["minlen"], cfg["hmult"], cfg["gap"]
    q = np.asarray(p8); n = len(q); D = round(dur, 3)
    isH = lambda e: is_habit_clip and e == (0.0, D)
    HD = {p: int(np.argmax(q[int(round(p[0] / fs)):max(int(round(p[0] / fs)) + 1, min(n, int(round(p[1] / fs)))), 1:].mean(0))) for p in evs}
    add = []
    for k in range(q.shape[1] - 1):
        for a, b in _runs(q[:, 1 + k] >= th, gap, fs):
            if (b - a) * fs < minlen: continue
            e = (round(a * fs, 3), round(min(b * fs, D), 3)); L = e[1] - e[0]
            if L <= 0 or _dup(e, evs + add): continue
            if any((min(e[1], p[1]) - max(e[0], p[0])) / (max(e[1], p[1]) - min(e[0], p[0])) >= 0.5 for p in evs): continue
            if [p for p in evs if not isH(p) and p[0] <= e[0] + 0.02 and p[1] >= e[1] - 0.02 and (p[1] - p[0]) >= hmult * L and HD[p] != k]:
                add.append(e)
    return sorted(evs + add)


def syn_shift(evs, dur, steps):
    """Synthetic edge shift, applied as the two steps that were shipped (net onset +8 ms, offset -6 ms)."""
    for s in steps:
        on, off = s["on"], s["off"]
        if s["clamp"]:
            evs = [(round(max(0.0, a + on), 3), round(min(dur, b + off), 3)) for a, b in evs if b + off > a + on]
        else:
            evs = [(round(a + on, 3), round(b + off, 3)) for a, b in evs if b + off > a + on]
    return evs


def run_chain(C, ids, nat, dur, arr, clips, habit_p, nat_members, syn_members, cfg,
              post_members=None, habit_hosts=None, return_stages=False):
    """Full decode chain. C: {tag: {"post": {cid: (T, 8)}, "dist": ...}}. habit_p: {cid: prob} for natural clips.
    post_members: members whose mean feeds bgext and type-C (default nat_members).
    habit_hosts: clips whose (0, D) event is excluded as a type-C host (default {c: habit_p[c] >= thr})."""
    nat_ro, syn_ro = load_readouts(cfg)
    syn = level_filter(decode(syn_members, syn_ro, ids, nat, dur, C, cfg["fallback_q"]), ids, nat, arr, clips, cfg["level_db"])
    seven = decode(nat_members, nat_ro, ids, nat, dur, C, cfg["fallback_q"])
    base = {c: (r3(seven[c]) if nat[c] else r3(syn[c])) for c in ids}
    post8 = E.average([C[t]["post"] for t in (post_members or nat_members)])
    bg = {c: (bgext(base[c], post8[c], dur[c], cfg["bgext"], cfg["fs"]) if nat[c] else base[c]) for c in ids}
    sh1 = {c: (bg[c] if nat[c] else syn_shift(bg[c], dur[c], cfg["syn_shift"][:1])) for c in ids}
    hab = {c: (habit_span(sh1[c], habit_p.get(c), dur[c], cfg["habit"]) if nat[c] else sh1[c]) for c in ids}
    if habit_hosts is None:
        habit_hosts = {c for c, p in habit_p.items() if p >= cfg["habit"]["thr"]}
    tc = {c: (typec_events(hab[c], post8[c], dur[c], c in habit_hosts, cfg["typec"], cfg["fs"]) if nat[c] else hab[c]) for c in ids}
    final = {c: (tc[c] if nat[c] else syn_shift(tc[c], dur[c], cfg["syn_shift"][1:])) for c in ids}
    if return_stages:
        return final, dict(champion=base, bgext=bg, shift1=sh1, habit=hab, typec=tc, final=final)
    return final


def write_predictions(preds, out_zip, ids):
    """Official format: zip holding one predictions.jsonl, one line per clip (events [] if none). Returns the inner md5."""
    SF.write_zip(preds, str(out_zip), set(ids))
    return hashlib.md5(zipfile.ZipFile(out_zip).read("predictions.jsonl")).hexdigest()


def read_predictions(zip_path):
    rows = {}
    for line in zipfile.ZipFile(zip_path).read("predictions.jsonl").decode().splitlines():
        o = json.loads(line)
        rows[o["clip_id"]] = [(e["onset"], e["offset"]) for e in o["events"]]
    return rows


def _union(evs):
    out = []
    for s, e in sorted(evs):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def agreement(a, b, collar=0.2):
    """How close two prediction sets {cid: [(on, off)]} are: clips differing, event F1 (greedy 1:1 match, onset AND
    offset within collar s) and pooled segment Dice on exact interval unions. Used to compare a rebuild with an upload."""
    tp = na = nb = diff = 0
    inter = tot = 0.0
    for c in b:
        A, B = [tuple(e) for e in a.get(c, [])], [tuple(e) for e in b[c]]
        diff += A != B; na += len(A); nb += len(B)
        used = set()
        for s, e in A:
            for j, (s2, e2) in enumerate(B):
                if j not in used and abs(s - s2) <= collar and abs(e - e2) <= collar:
                    used.add(j); tp += 1
                    break
        ua, ub = _union(A), _union(B)
        tot += sum(e - s for s, e in ua) + sum(e - s for s, e in ub)
        inter += sum(max(0.0, min(e, e2) - max(s, s2)) for s, e in ua for s2, e2 in ub)
    return {"clips_differing": diff, "clips": len(b), "events": (na, nb),
            "event_f1": 2 * tp / (na + nb) if na + nb else 1.0, "dice": 2 * inter / tot if tot else 1.0}
