"""The state of an editing session: which of the cleaner's proposed edits have been applied.

The cleaner proposes two kinds of edit (delete a wrong human note, insert a missing one). The
editor lets the user apply them a selection at a time, so a session is just the set of proposals
applied so far, plus a history so every step can be undone and redone. Nothing here touches a
file or a window: the cleaned MIDI is built on demand from the applied edits, and the original
files are never changed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import mido
import numpy as np

from .cleaning import ADD, REMOVE, CleaningPlan, build_cleaned_midi
from .comparison import ADDED, REMOVED, Comparison
from .loader import MidiData


@dataclass(frozen=True)
class EditState:
    removed: frozenset[int] = frozenset()  # indices of human notes that were deleted
    added: frozenset[int] = frozenset()  # indices into plan.add of the notes that were inserted

    @property
    def count(self) -> int:
        return len(self.removed) + len(self.added)


class EditSession:
    def __init__(self, perfect: MidiData, human: MidiData, plan: CleaningPlan, comparison: Comparison,
                 state: EditState | None = None) -> None:
        self.perfect, self.human, self.plan, self.base = perfect, human, plan, comparison
        self._source = plan.add_source if plan.add_source is not None else np.zeros(0, int)
        start = state or EditState()
        self._history: list[tuple[str, EditState]] = [("", start)]
        self._pos = 0
        self._saved = start

    # ---- what is applied --------------------------------------------------------

    @property
    def state(self) -> EditState:
        return self._history[self._pos][1]

    @property
    def dirty(self) -> bool:
        """Whether the applied edits differ from what was last saved in a project."""
        return self.state != self._saved

    def mark_saved(self) -> None:
        self._saved = self.state

    def load_state(self, state: EditState) -> None:
        """Start over from `state` (a reopened project): nothing before it can be undone."""
        self._history, self._pos, self._saved = [("", state)], 0, state

    def view(self) -> Comparison:
        """The comparison with the applied edits reflected in the notes' statuses."""
        removed, added = sorted(self.state.removed), sorted(self.state.added)
        human, perfect = self.base.human.status.copy(), self.base.perfect.status.copy()
        human[removed] = REMOVED
        human[self.base.n_human + np.array(added, dtype=int)] = ADDED
        perfect[self._source[np.array(added, dtype=int)]] = ADDED
        return replace(self.base, human=replace(self.base.human, status=human),
                       perfect=replace(self.base.perfect, status=perfect))

    # ---- editing ----------------------------------------------------------------

    def effect_of(self, perfect_notes: set[int], human_notes: set[int]) -> tuple[frozenset[int], frozenset[int]]:
        """(notes to delete, additions to insert) that cleaning these roll notes would newly apply.

        Wrong human notes are deleted, and a missing note is inserted whether the selection holds
        the source note it comes from or the proposed note itself. Anything else is left alone.
        """
        base, state, n_human = self.base, self.state, self.base.n_human
        remove = {h for h in human_notes
                  if h < n_human and base.human.status[h] == REMOVE and h not in state.removed}
        add = {h - n_human for h in human_notes if h >= n_human}
        add |= {int(base.perfect.partner[p]) - n_human for p in perfect_notes if base.perfect.status[p] == ADD}
        return frozenset(remove), frozenset(add - state.added)

    def clean(self, perfect_notes: set[int], human_notes: set[int]) -> str | None:
        """Apply the cleaner's edits to the selected notes; returns what was done, None if nothing."""
        remove, add = self.effect_of(perfect_notes, human_notes)
        if not remove and not add:
            return None
        parts = ([f"removed {len(remove):,} note(s)"] if remove else []) + \
                ([f"added {len(add):,} note(s)"] if add else [])
        label = "Clean selection: " + " and ".join(parts)
        new = EditState(self.state.removed | remove, self.state.added | add)
        del self._history[self._pos + 1:]  # a new edit forgets whatever could still be redone
        self._history.append((label, new))
        self._pos += 1
        return label

    @property
    def undo_label(self) -> str | None:
        return self._history[self._pos][0] if self._pos > 0 else None

    @property
    def redo_label(self) -> str | None:
        return self._history[self._pos + 1][0] if self._pos + 1 < len(self._history) else None

    def undo(self) -> str | None:
        label = self.undo_label
        if label is not None:
            self._pos -= 1
        return label

    def redo(self) -> str | None:
        label = self.redo_label
        if label is not None:
            self._pos += 1
        return label

    # ---- output -----------------------------------------------------------------

    def applied_plan(self) -> CleaningPlan:
        """A plan holding only the edits applied so far."""
        state = self.state
        return CleaningPlan(remove=np.array(sorted(state.removed), dtype=int),
                            add=[self.plan.add[k] for k in sorted(state.added)])

    def build_midi(self) -> mido.MidiFile:
        """The human file with the applied edits; every other event is kept exactly as recorded."""
        return build_cleaned_midi(self.applied_plan(), self.human)
