"""Score every human MIDI in MIDIs/ against every perfect source and report separation.

Ground truth comes from the folder layout: a human file belongs to the perfect source in
its own folder. Usage:  python tools/evaluate.py [--csv out.csv] [--jobs N]
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

from midi_cleaner.confidence import compare  # noqa: E402
from midi_cleaner.loader import load_midi  # noqa: E402

MIDI_DIR = ROOT / "MIDIs"
_cache: dict[Path, object] = {}


def discover() -> tuple[list[Path], list[Path]]:
    perfects, humans = [], []
    for f in sorted(MIDI_DIR.rglob("*")):
        if "Cleaned" in f.relative_to(MIDI_DIR).parts:  # output of the cleaner, not input data
            continue
        if f.suffix.lower() in (".mid", ".midi"):
            (perfects if "PERFECT SOURCE" in f.name else humans).append(f)
    return perfects, humans


def _load(path: Path):
    if path not in _cache:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _cache[path] = load_midi(path)
    return _cache[path]


def _score(job: tuple[Path, Path]) -> dict:
    perfect, human = job
    t = time.time()
    r = compare(_load(perfect), _load(human))
    return {"perfect": perfect.parent.name, "human": human.name, "truth": perfect.parent == human.parent,
            "seconds": time.time() - t, **asdict(r)}


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability a random true pair outscores a random false pair."""
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    perfects, humans = discover()
    jobs = [(p, h) for h in humans for p in perfects]
    print(f"{len(perfects)} perfect sources, {len(humans)} human files, {len(jobs)} comparisons")
    t0 = time.time()
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(_score, jobs, chunksize=8))
    print(f"done in {time.time() - t0:.0f}s (slowest pair {max(r['seconds'] for r in rows):.1f}s)\n")

    pos = np.array([r["confidence"] for r in rows if r["truth"]])
    neg = np.array([r["confidence"] for r in rows if not r["truth"]])
    print(f"TRUE pairs  n={len(pos):3d}  min={pos.min():.3f} p10={np.percentile(pos, 10):.3f} "
          f"median={np.median(pos):.3f}")
    print(f"FALSE pairs n={len(neg):3d}  max={neg.max():.3f} p99={np.percentile(neg, 99):.3f} "
          f"median={np.median(neg):.3f}")
    print(f"AUC={auc(pos, neg):.4f}   gap (min true - max false) = {pos.min() - neg.max():+.3f}")

    # Per human file: does its own piece win, and by how much?
    wrong = 0
    print("\nWorst true pairs, and every false pair scoring above 0.3:")
    for r in sorted((r for r in rows if r["truth"]), key=lambda r: r["confidence"])[:8]:
        print(f"  TRUE  {r['confidence']:.3f}  a={r['alignment_similarity']:.2f} R={r['raw_recall']:.2f} "
              f"P={r['raw_precision']:.2f} ch={r['chance_recall']:.2f} cov={r['coverage']:.2f} seg={r['segments']} nr={r['note_ratio']:.2f} "
              f"sr={r['span_ratio']:.2f} | {r['human'][:60]}")
    for r in sorted((r for r in rows if not r["truth"] and r["confidence"] > 0.3),
                    key=lambda r: -r["confidence"]):
        wrong += 1
        print(f"  FALSE {r['confidence']:.3f}  a={r['alignment_similarity']:.2f} R={r['raw_recall']:.2f} "
              f"P={r['raw_precision']:.2f} | perfect={r['perfect']} human={r['human'][:50]}")
    if not wrong:
        print("  (no false pair above 0.3)")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
