"""Build a labeled benchmark and record the raw matching evidence for offline tuning.

For every human file in MIDIs/ it creates:
  same piece (label 1)   intact, skipped section, extra repeat, first 70%, last 60%,
                         light note errors
  not the piece (label 0) every OTHER perfect source, transposed copies, heavily scrambled copy
and aligns each against the right perfect source once. The evidence is then re-scored at
several match tolerances, so scoring formulas can be compared without re-aligning.
Usage:  python tools/benchmark.py out.csv [--jobs N]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import stress as S  # noqa: E402
from evaluate import discover  # noqa: E402
from midi_cleaner import confidence as C  # noqa: E402
from midi_cleaner.alignment import align  # noqa: E402

TOLERANCES = (0.10, 0.15, 0.25, 0.40)


def variants(human, rng):
    """(name, label, MidiData) for damaged copies of one human file."""
    allnotes = np.ones(human.note_count, bool)
    yield "intact", 1, human
    yield "skip 20%", 1, S.cut(human, 0.4, 0.6)
    yield "repeat 15%", 1, S.repeat(human, 0.3, 0.45)
    yield "first 70%", 1, S.select(human, S.fraction(human, 0, 0.7))
    yield "last 60%", 1, S.select(human, S.fraction(human, 0.4, 1.01), shift=-0.4 * human.span)
    yield "10% wrong notes", 1, S.scramble(human, 0.10, rng)
    yield "transposed +1", 0, S.select(human, allnotes, pitch_shift=1)
    yield "transposed +12", 0, S.select(human, allnotes, pitch_shift=12)
    yield "60% wrong notes", 0, S.scramble(human, 0.60, rng)


def job(args: tuple[Path, Path, str, int]) -> list[dict]:
    perfect_path, human_path, kind, seed = args
    perfect, human = S.load(perfect_path), S.load(human_path)
    rng = np.random.default_rng(seed)
    if kind == "own":
        cases = list(variants(human, rng))
    else:  # a different piece's perfect source: always a mismatch
        cases = [("other piece", 0, human)]
    rows = []
    for name, label, data in cases:
        coarse, fine = align(perfect, data)
        row = {"perfect": perfect_path.parent.name, "human": human_path.name, "variant": name, "label": label}
        for tol in TOLERANCES:
            C.MATCH_TOLERANCE = tol
            r = asdict(C.score(perfect, data, coarse, fine))
            if tol == TOLERANCES[0]:
                row.update({k: r[k] for k in ("coarse_similarity", "alignment_similarity", "coverage",
                                              "segments", "note_ratio", "span_ratio")})
            for key in ("raw_recall", "raw_precision", "chance_recall", "chance_precision"):
                row[f"{key}@{tol}"] = r[key]
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    perfects, humans = discover()
    jobs = []
    for n, h in enumerate(humans):
        for p in perfects:
            jobs.append((p, h, "own" if p.parent == h.parent else "other", n))
    print(f"{len(jobs)} alignment jobs")
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with ProcessPoolExecutor(args.jobs) as pool:
            rows = [r for chunk in pool.map(job, jobs, chunksize=4) for r in chunk]
    print(f"{len(rows)} labeled rows in {time.time() - t0:.0f}s")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
