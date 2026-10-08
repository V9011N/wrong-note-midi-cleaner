"""Clean every human MIDI in MIDIs/ against its piece's perfect source (development helper).

Writes  MIDIs/<piece>/Cleaned/<name> - CLEANED.mid  and checks each result:
  * every non-note event (pedal, tempo, ...) is identical to the original;
  * every note we did not touch is identical; removed notes are gone; added notes are present;
  * re-matching the cleaned file against the source scores at least as well.
Usage:  python tools/clean_all.py [--only SUBSTRING] [--jobs N]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mido
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluate import discover  # noqa: E402  (tools/ is on sys.path when run as a script)
from midi_cleaner.alignment import align  # noqa: E402
from midi_cleaner.cleaning import apply_cleaning, plan_cleaning  # noqa: E402
from midi_cleaner.confidence import score  # noqa: E402
from midi_cleaner.loader import load_midi  # noqa: E402


def non_note_events(path: Path) -> list[list[tuple]]:
    midi = mido.MidiFile(str(path), clip=True)
    out = []
    for track in midi.tracks:
        tick, events = 0, []
        for msg in track:
            tick += msg.time
            if msg.type not in ("note_on", "note_off"):
                events.append((tick, msg.type, tuple(sorted(msg.dict().items()))[:0], str(msg.copy(time=0))))
        out.append(events)
    return out


def drum_events(path: Path) -> list[tuple]:
    midi = mido.MidiFile(str(path), clip=True)
    return [(i, msg.type) for i, t in enumerate(midi.tracks) for msg in t
            if msg.type in ("note_on", "note_off") and msg.channel == 9]


def clean_one(job: tuple[Path, Path]) -> dict:
    perfect_path, human_path = job
    warnings.simplefilter("ignore")
    perfect, human = load_midi(perfect_path), load_midi(human_path)
    coarse, fine = align(perfect, human)
    before = score(perfect, human, coarse, fine)
    plan = plan_cleaning(perfect, human, fine)

    out_dir = human_path.parent / "Cleaned"
    out_path = out_dir / f"{human_path.stem} - CLEANED.mid"
    apply_cleaning(plan, human, out_path)

    cleaned = load_midi(out_path)
    problems = []
    if non_note_events(human_path) != non_note_events(out_path):
        problems.append("non-note events changed")
    if drum_events(human_path) != drum_events(out_path):
        problems.append("drum-channel events changed")

    keep = np.ones(human.note_count, bool)
    keep[plan.remove] = False
    expected = ([(int(p), int(v), float(o)) for p, v, o in
                 zip(human.pitches[keep], human.velocities[keep], human.onsets[keep])]
                + [(n.pitch, n.velocity, n.onset) for n in plan.add])
    got = list(zip(cleaned.pitches.tolist(), cleaned.velocities.tolist(), cleaned.onsets.tolist()))
    if len(expected) != len(got):
        problems.append(f"note count {len(got)} != expected {len(expected)}")
    else:  # per (pitch, velocity): the sorted onsets must agree to within a tick or two
        groups: dict[tuple, list[list[float]]] = {}
        for p, v, o in expected:
            groups.setdefault((p, v), [[], []])[0].append(o)
        for p, v, o in got:
            groups.setdefault((p, v), [[], []])[1].append(o)
        worst = 0.0
        for e, g in groups.values():
            if len(e) != len(g):
                worst = np.inf
                break
            if e:
                worst = max(worst, float(np.abs(np.sort(e) - np.sort(g)).max()))
        if worst > 0.005:
            problems.append(f"note content differs from plan (max onset error {worst:.4f}s)")

    coarse2, fine2 = align(perfect, cleaned)
    after = score(perfect, cleaned, coarse2, fine2)
    return {"piece": human_path.parent.name, "human": human_path.name, "n": human.note_count,
            "remove": plan.n_remove, "add": plan.n_add, "displaced": plan.displaced,
            "untrusted": plan.untrusted, "before": before, "after": after, "problems": problems}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    perfects, humans = discover()
    jobs = [(p, h) for h in humans for p in perfects
            if p.parent == h.parent and args.only.lower() in h.name.lower()]
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(clean_one, jobs, chunksize=1))

    print(f"{'piece / file':62s} {'notes':>6s} {'-rm':>5s} {'+add':>5s} {'disp':>5s} {'untr':>5s}"
          f"  recall  precision  confidence")
    for r in rows:
        b, a = r["before"], r["after"]
        print(f"{(r['piece'][:20] + ' / ' + r['human'][-38:]):62s} {r['n']:6d} {r['remove']:5d} {r['add']:5d} "
              f"{r['displaced']:5d} {r['untrusted']:5d}  {b.raw_recall:.2f}>{a.raw_recall:.2f}  "
              f"{b.raw_precision:.2f}>{a.raw_precision:.2f}  {b.confidence:.2f}>{a.confidence:.2f}"
              + ("   PROBLEMS: " + "; ".join(r["problems"]) if r["problems"] else ""))
    tot = sum(r["n"] for r in rows)
    print(f"\n{len(rows)} files; removed {sum(r['remove'] for r in rows)} / added {sum(r['add'] for r in rows)} "
          f"notes of {tot} ({sum(r['remove'] for r in rows) / tot:.1%} / {sum(r['add'] for r in rows) / tot:.1%}); "
          f"{sum(bool(r['problems']) for r in rows)} files with integrity problems")


if __name__ == "__main__":
    main()
