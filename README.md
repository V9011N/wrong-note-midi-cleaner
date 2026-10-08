# wrong-note-midi-cleaner

**Fix the wrong notes in a human piano performance by comparing it against the score it was
supposed to play.**

This tool takes **two** MIDI files of the **same piece**:

| | |
|---|---|
| **Perfect source** | a clean MIDI of the written music (e.g. exported from notation software) |
| **Human MIDI** | a recorded performance of that piece (e.g. a MAESTRO performance) |

It works out which notes in the performance the composer never wrote (stray touches, brushed
neighbour keys, wrong notes) and which written notes the performer missed, then **removes the
extra notes and inserts the missing ones** in the human file, leaving everything else of the
performance exactly as it was played.

## What this is *not*

There are many MIDI cleaners. They look at **one** file and apply generic rules: quantize,
snap to a grid, delete notes shorter than X ms or quieter than Y, merge duplicates, trim
silence. **This is not one of those.**

- It does **not** guess what is "noise" from the performance alone. A note is only removed if
  the *score* does not contain it, and only added if the *score* does.
- It does **not** quantize, retime, normalise velocities, change note lengths, or touch the
  pedal. The performer's timing, dynamics and articulation are the thing being preserved.
- It cannot run on a single file. Without the matching perfect source there is nothing to
  compare against, and it will refuse to edit when the two files do not look like the same piece.

In short: a loud, perfectly timed wrong note and a quiet, perfectly intended ornament look the
same to a generic cleaner. Against the score they are easy to tell apart.

## What it changes

Exactly two kinds of edit, each of which can be switched off:

- **Remove notes**: performed notes that have no counterpart in the perfect source.
- **Add notes**: written notes the performance lacks. An added note takes its onset from the
  chord it belongs to or from the performer's own neighbouring notes, its velocity from nearby
  notes the performer played, and its length from the source scaled to the local tempo.

Everything else in the human file is written back unchanged: every other note's timing and
velocity, pedal and other controller data, tempo events, tracks. The original file is never
overwritten.

Not done yet (deliberately deferred): evening out notes inside scales and other fast passages.

## How it works

1. **Is this the right pair?** The two files live on different clocks (the score has its own
   tempo map, the performance has rubato), so raw timestamps are never compared. The files are
   aligned with dynamic time warping, and a 0-100% **match confidence** is computed from how
   well the notes agree beyond what chance would give. Cleaning is disabled for files that do
   not look like the same piece.
2. **Alignment that survives real performances.** A coarse pass over the whole piece can *jump*,
   so a take that starts mid-piece, skips a section or replays one still aligns (reported as
   *coverage* and *aligned stretches*). A fine pass then aligns each stretch to roughly
   note precision.
3. **Note matching.** Human notes are matched one-to-one to source notes of the same pitch
   (exact dynamic programming, so repeated notes and tremolos pair correctly).
4. **Conservative edits.** A leftover performed note and a leftover source note of the same
   pitch that are close in time are one note played a little off, not a wrong note, so they are
   left alone. Nothing is edited near the joins between aligned stretches or where the local
   alignment is poorly supported.

## Using it

```bash
pip install -r requirements.txt
python app.py
```

1. Choose the **perfect source** and the **human MIDI**. The match confidence appears
   automatically.
2. The **Add notes** / **Remove notes** boxes are ticked from what was found. You may change
   them, but at least one must stay selected.
3. Click **Clean**. When cleaning is done a save dialog opens so you can choose where the
   result goes.

## Measured so far

On 52 human performances of 10 pieces (Beethoven, Chopin, Debussy, Liszt, Rachmaninoff,
Scriabin):

- **Matching:** every true pair scored 0.76-0.99 confidence; no mismatched pair scored above
  0.08. Skipped sections, extra repeats and partial takes are detected and reported.
- **Ground-truth cleaning test** (3% of correct notes deleted and as many realistic stray
  touches injected, then cleaned): **92%** of strays removed, **82%** of deleted notes
  restored (restored note onsets a median **12 ms** from the original), about **0.1%** of
  correct notes wrongly removed.
- **Integrity:** on all 52 files every non-note event and every untouched note is identical
  to the original.

**Known limits.** On untouched performances it proposes removing about 6% and adding about 5% of
the notes. The removed notes are quieter and shorter than average, which fits stray touches, but
there is no ground truth proving each one is a mistake rather than something the performer
really played that the score omits. Listen before trusting a result. Very noisy takes (about 15%
wrong notes or more) on dense pieces fall into the "uncertain" band.

## Repository layout

```
app.py                  tkinter UI
midi_cleaner/
  loader.py             MIDI -> note table (mido); remembers where every note came from
  alignment.py          coarse + fine time alignment (jump-aware DTW)
  confidence.py         match confidence and verdict
  cleaning.py           note matching, edit planning, writing the cleaned file
tools/
  evaluate.py           score every human file against every perfect source
  stress.py             structural damage scenarios (skips, repeats, partial takes ...)
  benchmark.py          labeled evidence for tuning the confidence formula
  clean_all.py          clean every file in MIDIs/ and verify integrity
  inject_test.py        ground-truth test: inject known errors, measure recovery
MIDIs/                  your data, not in the repo (see below)
```

### Data layout used by the tools

The `MIDIs/` folder is not committed (it holds third-party material such as the MAESTRO
dataset, which has its own licence). The tools expect one folder per piece:

```
MIDIs/<piece>/<anything> - PERFECT SOURCE.mid     the score
MIDIs/<piece>/<performance>.midi                  one or more human performances
MIDIs/<piece>/Cleaned/                            written by tools/clean_all.py
```

Any file not named `PERFECT SOURCE` is treated as a human performance of its folder's piece;
that folder layout is the ground truth the evaluation tools use.
