"""Ground-truth test for cleaning: damage real human files in known ways, then see what is fixed.

For each human file we
  1. plan cleaning on the untouched file (the *baseline* edits: real differences from the score);
  2. delete some confirmed-correct notes and inject realistic stray touches (short, quiet-ish
     notes a semitone or two from a real note, played at about the same moment);
  3. plan cleaning on the damaged file and score it against what we did:
       stray removal recall   share of injected strays that get removed
       deleted-note recall    share of deleted notes that get put back (right pitch, +/-100 ms)
       collateral             edits beyond the baseline and beyond the injected damage
Usage:  python tools/inject_test.py [--jobs N] [--rate 0.03]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluate import discover  # noqa: E402
from midi_cleaner.alignment import align  # noqa: E402
from midi_cleaner.cleaning import CleaningPlan, NewNote, apply_cleaning, plan_cleaning  # noqa: E402
from midi_cleaner.loader import MidiData, load_midi  # noqa: E402


def keys_of(data: MidiData, idx: np.ndarray) -> list[tuple[int, float]]:
    return [(int(data.pitches[i]), float(data.onsets[i])) for i in idx]


def contains(pool: dict[int, np.ndarray], pitch: int, onset: float, tol: float) -> bool:
    arr = pool.get(pitch)
    return arr is not None and len(arr) > 0 and bool(np.abs(arr - onset).min() <= tol)


def by_pitch(items: list[tuple[int, float]]) -> dict[int, np.ndarray]:
    out: dict[int, list[float]] = {}
    for p, t in items:
        out.setdefault(p, []).append(t)
    return {p: np.array(v) for p, v in out.items()}


def run(job: tuple[Path, Path, float, int]) -> dict:
    perfect_path, human_path, rate, seed = job
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(seed)
    perfect, human = load_midi(perfect_path), load_midi(human_path)
    _, fine = align(perfect, human)
    base = plan_cleaning(perfect, human, fine)

    # Candidates for damage: notes the cleaner did not itself flag, i.e. believed correct.
    ok = np.ones(human.note_count, bool)
    ok[base.remove] = False
    ok_idx = np.flatnonzero(ok)
    n_dmg = int(rate * human.note_count)
    deleted = rng.choice(ok_idx, n_dmg, replace=False)
    anchors = rng.choice(ok_idx, n_dmg, replace=False)
    strays = []
    for a in anchors:
        pitch = int(np.clip(human.pitches[a] + rng.choice([-2, -1, 1, 2]), 21, 108))
        strays.append(NewNote(pitch, float(human.onsets[a] + rng.normal(0, 0.02)), float(rng.uniform(0.03, 0.12)),
                              int(rng.integers(15, 75)), int(human.tracks[a]), int(human.channels[a])))

    with tempfile.TemporaryDirectory() as tmp:
        damaged_path = Path(tmp) / "damaged.mid"
        apply_cleaning(CleaningPlan(remove=np.sort(deleted), add=strays), human, damaged_path)
        damaged = load_midi(damaged_path)
        _, fine_d = align(perfect, damaged)
        plan = plan_cleaning(perfect, damaged, fine_d)

    removed = by_pitch(keys_of(damaged, plan.remove))
    added = by_pitch([(n.pitch, n.onset) for n in plan.add])
    base_removed = by_pitch(keys_of(human, base.remove))
    base_added = by_pitch([(n.pitch, n.onset) for n in base.add])
    injected = by_pitch([(n.pitch, n.onset) for n in strays])
    deleted_keys = by_pitch(keys_of(human, deleted))

    stray_hit = np.mean([contains(removed, n.pitch, n.onset, 0.003) for n in strays])
    del_hit = np.mean([contains(added, p, t, 0.10) for p, t in keys_of(human, deleted)])
    errs = []
    for p_, t_ in keys_of(human, deleted):
        near = [abs(n.onset - t_) for n in plan.add if n.pitch == p_ and abs(n.onset - t_) <= 0.5]
        if near:
            errs.append(min(near))
    extra_rm = sum(1 for p, t in keys_of(damaged, plan.remove)
                   if not contains(injected, p, t, 0.003) and not contains(base_removed, p, t, 0.003))
    extra_add = sum(1 for n in plan.add
                    if not contains(deleted_keys, n.pitch, n.onset, 0.10)
                    and not contains(base_added, n.pitch, n.onset, 0.10))
    # Did we ever delete an *intended* original note that was not one of the injected strays?
    harmed = sum(1 for p, t in keys_of(damaged, plan.remove)
                 if not contains(injected, p, t, 0.003) and not contains(base_removed, p, t, 0.003)
                 and contains(by_pitch(keys_of(human, ok_idx)), p, t, 0.003))
    return {"piece": human_path.parent.name, "human": human_path.name, "n": human.note_count, "damage": n_dmg,
            "stray_recall": float(stray_hit), "deleted_recall": float(del_hit),
            "time_errors": errs, "extra_remove": extra_rm, "extra_add": extra_add, "harmed": harmed,
            "baseline_remove": base.n_remove, "baseline_add": base.n_add}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--rate", type=float, default=0.03)
    ap.add_argument("--per-piece", type=int, default=2)
    args = ap.parse_args()
    perfects, humans = discover()
    jobs, seen = [], {}
    for h in humans:
        for p in perfects:
            if p.parent == h.parent and seen.get(h.parent, 0) < args.per_piece:
                seen[h.parent] = seen.get(h.parent, 0) + 1
                jobs.append((p, h, args.rate, len(jobs)))
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(run, jobs, chunksize=1))

    print(f"damage rate {args.rate:.0%}: that many correct notes deleted and strays injected per file\n")
    print(f"{'piece':26s} {'dmg':>5s}  stray-removed  deleted-restored  extra-rm extra-add  (baseline -rm/+add)")
    for r in rows:
        print(f"{r['piece'][:26]:26s} {r['damage']:5d}  {r['stray_recall']:12.0%}  {r['deleted_recall']:15.0%}"
              f"  {r['extra_remove']:8d} {r['extra_add']:9d}   ({r['baseline_remove']}/{r['baseline_add']})")
    tot = sum(r["damage"] for r in rows)
    wavg = lambda k: sum(r[k] * r["damage"] for r in rows) / tot  # noqa: E731
    errs = np.array([e for r in rows for e in r["time_errors"]])
    print(f"\nrestored-note timing error: median {np.median(errs) * 1000:.0f} ms, "
          f"90th percentile {np.percentile(errs, 90) * 1000:.0f} ms, over {len(errs)} notes")
    print(f"OVERALL  strays removed {wavg('stray_recall'):.1%}   deleted notes restored {wavg('deleted_recall'):.1%}   "
          f"extra removals {sum(r['extra_remove'] for r in rows)}   extra additions {sum(r['extra_add'] for r in rows)}"
          f"   original notes wrongly removed {sum(r['harmed'] for r in rows)}")


if __name__ == "__main__":
    main()
