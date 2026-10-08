"""Stress the matcher with structurally damaged copies of real human performances.

Each scenario edits a real human MIDI (cut a section, repeat a section, splice in another
piece, transpose, scramble notes ...) and scores it against its true perfect source.
`expect` says what a sensible tool should conclude:
  match    clearly the same piece  -> confidence should stay high
  partial  same piece, only part of it played -> identity confidence stays up, coverage < 90%
  weak     badly corrupted -> must not read as a clean match
  other    NOT this piece -> confidence should be low
Usage:  python tools/stress.py
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from midi_cleaner.confidence import compare  # noqa: E402
from midi_cleaner.loader import MidiData, load_midi  # noqa: E402

MIDI_DIR = ROOT / "MIDIs"
CASES = [  # (folder, human file substring) - a spread of short/long, calm/dense pieces
    ("Chopin Etude Op 10 No 4", "2011 - Frédéric Chopin Etude Op. 10 No. 4 in C-Sharp Minor - 01"),
    ("Debussy L'isle Joyeuse", "2013 - Claude Debussy L'isle joyeuse - 01"),
    ("Scriabin Sonata 5", "2014 - Alexander Scriabin Sonata No. 5, Op. 53 - 01"),
    ("Rach Moments Musicaux 4", "2011 - Sergei Rachmaninoff"),
]


def perfect_of(folder: str) -> MidiData:
    return load(next((MIDI_DIR / folder).glob("*PERFECT SOURCE*")))


def human_of(folder: str, key: str) -> MidiData:
    return load(next(f for f in (MIDI_DIR / folder).glob("*.mid*") if key in f.name))


def load(path: Path) -> MidiData:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_midi(path)


def select(d: MidiData, mask: np.ndarray, shift: np.ndarray | float = 0.0, pitch_shift: int = 0) -> MidiData:
    return replace(d, onsets=d.onsets[mask] + (shift[mask] if isinstance(shift, np.ndarray) else shift),
                   offsets=d.offsets[mask] + (shift[mask] if isinstance(shift, np.ndarray) else shift),
                   pitches=d.pitches[mask] + pitch_shift, velocities=d.velocities[mask])


def fraction(d: MidiData, lo: float, hi: float) -> np.ndarray:
    """Mask of notes whose onset lies in [lo, hi) as fractions of the performance."""
    t = (d.onsets - d.start) / max(d.span, 1e-9)
    return (t >= lo) & (t < hi)


def cut(d: MidiData, lo: float, hi: float) -> MidiData:
    """Remove the [lo, hi) fraction and close the gap."""
    keep = ~fraction(d, lo, hi)
    gap = (hi - lo) * d.span
    return select(d, keep, shift=np.where(fraction(d, hi, 1.01), -gap, 0.0))


def repeat(d: MidiData, lo: float, hi: float) -> MidiData:
    """Play the [lo, hi) section twice."""
    seg = fraction(d, lo, hi)
    length = (hi - lo) * d.span
    later = d.onsets - d.start >= hi * d.span
    first = select(d, ~later)
    again = select(d, seg, shift=length)
    rest = select(d, later, shift=length)
    return merge(first, again, rest)


def merge(*parts: MidiData) -> MidiData:
    on = np.concatenate([p.onsets for p in parts])
    order = np.argsort(on, kind="stable")
    cat = lambda name: np.concatenate([getattr(p, name) for p in parts])[order]  # noqa: E731
    return replace(parts[0], onsets=on[order], offsets=cat("offsets"), pitches=cat("pitches"),
                   velocities=cat("velocities"))


def splice(a: MidiData, b: MidiData, at: float) -> MidiData:
    """First `at` of a, then the remainder of b (a different piece)."""
    head = select(a, fraction(a, 0.0, at))
    tail_mask = fraction(b, at, 1.01)
    end_a = at * a.span
    tail = select(b, tail_mask, shift=a.start + end_a - (b.start + at * b.span))
    return merge(head, tail)


def scramble(d: MidiData, rate: float, rng: np.random.Generator) -> MidiData:
    pitches = d.pitches.copy()
    hit = rng.random(len(pitches)) < rate
    pitches[hit] = rng.integers(30, 95, hit.sum())
    return replace(d, pitches=pitches)


def scenarios(perfect: MidiData, human: MidiData, other: MidiData, rng):
    yield "intact", "match", human
    yield "first 70% only", "partial", select(human, fraction(human, 0, 0.7))
    yield "last 60% only", "partial", select(human, fraction(human, 0.4, 1.01),
                                             shift=-0.4 * human.span)
    yield "middle 40% only", "partial", select(human, fraction(human, 0.3, 0.7), shift=-0.3 * human.span)
    yield "skipped 20% section", "match", cut(human, 0.4, 0.6)
    yield "skipped 35% section", "partial", cut(human, 0.3, 0.65)
    yield "extra repeat of 15%", "match", repeat(human, 0.3, 0.45)
    yield "extra repeat of 30%", "match", repeat(human, 0.2, 0.5)
    yield "half this + half other piece", "weak", splice(human, other, 0.5)
    yield "transposed +1", "other", select(human, np.ones(human.note_count, bool), pitch_shift=1)
    yield "transposed +12", "other", select(human, np.ones(human.note_count, bool), pitch_shift=12)
    yield "15% of notes scrambled", "match", scramble(human, 0.15, rng)
    yield "30% of notes scrambled", "weak", scramble(human, 0.3, rng)
    yield "60% of notes scrambled", "other", scramble(human, 0.6, rng)
    yield "OTHER piece entirely", "other", other


def main() -> None:
    rng = np.random.default_rng(1)
    humans = {c: (perfect_of(c[0]), human_of(*c)) for c in CASES}
    for case, (perfect, human) in humans.items():
        other = next(h for c, (_, h) in humans.items() if c != case)
        print(f"\n== {case[0]}  (perfect {perfect.note_count} notes, {perfect.span:.0f}s)")
        for name, expect, variant in scenarios(perfect, human, other, rng):
            r = compare(perfect, variant)
            flag = {"match": "ok " if r.confidence >= 0.60 else "LOW",
                    "partial": "ok " if r.confidence >= 0.30 and r.coverage < 0.9 else "?? ",
                    "weak": "ok " if (r.confidence < 0.60 or "(" in r.verdict) else "HI ",
                    "other": "ok " if r.confidence < 0.30 else "HI "}[expect]
            print(f"  {flag} {name:30s} expect={expect:7s} conf={r.confidence:.2f}  a={r.alignment_similarity:.2f} "
                  f"R={r.raw_recall:.2f} P={r.raw_precision:.2f} ch={r.chance_recall:.2f} cov={r.coverage:.2f} seg={r.segments}")


if __name__ == "__main__":
    main()
