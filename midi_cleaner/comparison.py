"""Everything the comparison view needs to draw, free of any GUI code.

The two files live on different clocks, so to stack them the perfect source is laid onto the
human performance's timeline using the alignment. A matched pair then sits roughly one above the
other, and the distance between the two ends of its line is the real timing difference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .alignment import Alignment
from .cleaning import ADD, DISPLACED, HELD, MATCHED, OUTSIDE, REMOVE, UNTRUSTED, CleaningPlan
from .loader import MidiData

SCALE_RANGE = (0.4, 2.5)  # a note's length on the shared timeline vs. in its own file
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


@dataclass(frozen=True)
class Roll:
    """One piano roll: every array is indexed like the notes of its MidiData.

    The human roll continues past the file's own notes with the notes the cleaner would add
    (status ADD); they are not in the file, only a proposal, and point at the perfect note
    they come from.
    """

    pitches: np.ndarray
    start: np.ndarray  # seconds on the shared timeline
    end: np.ndarray
    file_time: np.ndarray  # seconds into its own file (from its first note)
    status: np.ndarray  # cleaning outcome of each note (cleaning.MATCHED, REMOVE, ...)
    partner: np.ndarray  # index of the matching note in the other roll, -1 if none


@dataclass(frozen=True)
class Comparison:
    perfect: Roll
    human: Roll
    t_min: float
    t_max: float
    pitch_lo: int
    pitch_hi: int


def note_name(pitch: int) -> str:
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def _warp(t: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Piecewise-linear map through (xs, ys), continuing at unit slope beyond the ends."""
    out = np.interp(t, xs, ys)
    out = np.where(t < xs[0], ys[0] + (t - xs[0]), out)
    return np.where(t > xs[-1], ys[-1] + (t - xs[-1]), out)


def _perfect_on_human_clock(pt: np.ndarray, p_partner: np.ndarray, seg_of_h: np.ndarray,
                            alignment: Alignment) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where each perfect onset falls on the human timeline, plus the anchors that gave it."""
    x = np.full(len(pt), np.nan)
    paired = np.flatnonzero(p_partner >= 0)
    seg_of_p = seg_of_h[p_partner[paired]]  # a replayed passage maps through the take that played it
    for k, seg in enumerate(alignment.segments):
        sel = paired[seg_of_p == k]
        x[sel] = np.interp(pt[sel], seg.p_knots, seg.h_of_p)
    for seg in alignment.segments:
        sel = np.flatnonzero(np.isnan(x) & (pt >= seg.p_lo) & (pt <= seg.p_hi))
        x[sel] = np.interp(pt[sel], seg.p_knots, seg.h_of_p)

    placed = ~np.isnan(x)
    if not placed.any():  # nothing aligned at all: show the files on their own clocks
        return pt.copy(), pt.copy(), pt.copy()
    order = np.argsort(pt[placed], kind="stable")
    anchors_p, anchors_x = pt[placed][order], x[placed][order]
    x[~placed] = _warp(pt[~placed], anchors_p, anchors_x)  # skipped stretches: between neighbours
    return x, anchors_p, anchors_x


def build_comparison(perfect: MidiData, human: MidiData, alignment: Alignment,
                     plan: CleaningPlan) -> Comparison:
    ht = human.onsets - human.start
    pt = perfect.onsets - perfect.start
    h_status, h_partner = plan.human_status, plan.human_partner
    p_status, p_partner = plan.perfect_status, plan.perfect_partner

    seg_of_h, _ = alignment.segment_of_human(ht)
    p_on, anchors_p, anchors_x = _perfect_on_human_clock(pt, p_partner, seg_of_h, alignment)
    scale = np.clip(_warp(pt + 0.5, anchors_p, anchors_x) - _warp(pt - 0.5, anchors_p, anchors_x),
                    *SCALE_RANGE)
    p_off = p_on + np.maximum(perfect.offsets - perfect.onsets, 0.0) * scale

    h_rel_off = human.offsets - human.start

    # Proposed additions follow the real notes in the human roll, each paired with its source.
    n_real = len(ht)
    source = plan.add_source if plan.add_source is not None else np.zeros(0, int)
    add_pitch = np.array([n.pitch for n in plan.add], dtype=int)
    add_on = np.array([n.onset for n in plan.add], dtype=float) - human.start
    add_off = add_on + np.array([n.duration for n in plan.add], dtype=float)
    p_partner = p_partner.copy()
    p_partner[source] = n_real + np.arange(len(source))

    h_pitches = np.concatenate([human.pitches, add_pitch])
    h_on = np.concatenate([ht, add_on])
    h_off = np.concatenate([h_rel_off, add_off])
    h_status = np.concatenate([h_status, np.full(len(source), ADD, h_status.dtype)])
    h_partner = np.concatenate([h_partner, source])

    pitch_lo = int(min(perfect.pitches.min(), h_pitches.min()))
    pitch_hi = int(max(perfect.pitches.max(), h_pitches.max()))
    return Comparison(
        perfect=Roll(perfect.pitches, p_on, p_off, pt, p_status, p_partner),
        human=Roll(h_pitches, h_on, h_off, h_on, h_status, h_partner),
        t_min=float(min(p_on.min(), h_on.min(), 0.0)),
        t_max=float(max(p_off.max(), h_off.max())),
        pitch_lo=pitch_lo,
        pitch_hi=max(pitch_hi, pitch_lo + 11),  # never less than an octave tall
    )


# ---- explaining a note -----------------------------------------------------------

HUMAN_MEANING = {
    ADD: "to be added: the score has this note and the performance lacks it "
         "(Add notes would insert it here, this is only a proposal)",
    MATCHED: "correct, it is in the perfect source",
    DISPLACED: "the right note, but its timing is well off the score",
    REMOVE: "wrong, it is not in the perfect source (Remove notes would delete it)",
    UNTRUSTED: "no match found, but the alignment here is too uncertain to call it wrong",
    OUTSIDE: "outside the part of the piece that could be aligned, so it was not judged",
}
PERFECT_MEANING = {
    MATCHED: "the human played it",
    DISPLACED: "the human played it, but well off the score's timing",
    ADD: "missing from the performance (Add notes would insert it where the blue note is)",
    HELD: "no match found, but it could not be placed safely, so it was left alone",
    UNTRUSTED: "no match found, but the alignment here is too uncertain to call it missing",
    OUTSIDE: "outside the part of the piece that could be aligned, so it was not judged",
}


def describe_note(comp: Comparison, side: str, index: int) -> str:
    """One sentence on a note and what it matched ('h' for human, 'p' for perfect)."""
    own, other = (comp.human, comp.perfect) if side == "h" else (comp.perfect, comp.human)
    who, other_who = ("Human", "perfect") if side == "h" else ("Perfect", "human")
    meaning = (HUMAN_MEANING if side == "h" else PERFECT_MEANING)[int(own.status[index])]
    text = f"{who} {note_name(int(own.pitches[index]))} at {own.file_time[index]:.2f} s: {meaning}."
    partner = int(own.partner[index])
    if partner >= 0 and int(own.status[index]) != ADD:
        h, p = (index, partner) if side == "h" else (partner, index)
        lag = comp.human.start[h] - comp.perfect.start[p]
        text += (f" Paired with the {other_who} note at {other.file_time[partner]:.2f} s; "
                 f"the human is {abs(lag) * 1000:.0f} ms {'late' if lag > 0 else 'early'} "
                 "once the tempo is accounted for.")
    return text
