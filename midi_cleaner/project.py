"""Save and reopen an editing session as a small JSON project file.

A project does not contain any MIDI. It names the two files, remembers a hash of each so a
changed file can be noticed, and lists which proposed edits were applied. Notes are identified
by where they sit in their file (track and event position), not by their index in the sorted
note table, so the saved edits still land on the right notes when the project is reopened and the
files are analysed again.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .editing import EditSession, EditState

PROJECT_EXTENSION = ".wnproject"
FORMAT = "wrong-note-midi-cleaner-project"
VERSION = 1


class ProjectError(Exception):
    """Raised with a user-presentable message when a project file can't be used."""


@dataclass(frozen=True)
class Project:
    perfect_path: Path
    human_path: Path
    perfect_hash: str
    human_hash: str
    removed: tuple[tuple[int, int], ...]  # (track, event position) of each deleted human note
    added: tuple[tuple[int, int], ...]  # (track, event position) of the source note of each insertion

    def changed_files(self) -> list[str]:
        """Names of the files that no longer match what the project was saved against."""
        return [role for role, path, saved in (("Perfect source", self.perfect_path, self.perfect_hash),
                                               ("Human MIDI", self.human_path, self.human_hash))
                if file_hash(path) != saved]

    def missing_files(self) -> list[Path]:
        return [p for p in (self.perfect_path, self.human_path) if not p.is_file()]


def file_hash(path: Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


# ---- writing ------------------------------------------------------------------------

def _file_entry(path: Path, project_dir: Path) -> dict:
    try:
        relative = os.path.relpath(path, project_dir)
    except ValueError:  # another drive on Windows
        relative = None
    return {"path": str(path), "relative": relative, "sha256": file_hash(path)}


def save_project(path: str | Path, session: EditSession) -> Path:
    """Write the session's applied edits to `path`; the file is replaced only once fully written."""
    path = Path(path)
    state, perfect, human = session.state, session.perfect, session.human
    document = {
        "format": FORMAT,
        "version": VERSION,
        "perfect": _file_entry(perfect.path.resolve(), path.resolve().parent),
        "human": _file_entry(human.path.resolve(), path.resolve().parent),
        "edits": {
            "remove": sorted([int(human.tracks[h]), int(human.on_events[h])] for h in state.removed),
            "add": sorted([int(perfect.tracks[p]), int(perfect.on_events[p])]
                          for p in (int(session.plan.add_source[k]) for k in state.added)),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(document, indent=2), encoding="utf-8")
    temp.replace(path)
    return path


# ---- reading ------------------------------------------------------------------------

def _locate(entry: object, project_dir: Path, role: str) -> tuple[Path, str]:
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise ProjectError(f"The project does not say where the {role} file is.")
    candidates = [Path(entry["path"])]
    if isinstance(entry.get("relative"), str):
        candidates.append((project_dir / entry["relative"]).resolve())  # the folder may have moved
    found = next((c for c in candidates if c.is_file()), candidates[0])
    return found, str(entry.get("sha256", ""))


def _pairs(values: object, what: str) -> tuple[tuple[int, int], ...]:
    try:
        return tuple((int(a), int(b)) for a, b in values)  # type: ignore[union-attr]
    except (TypeError, ValueError) as exc:
        raise ProjectError(f"The project's list of {what} is damaged.") from exc


def load_project(path: str | Path) -> Project:
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProjectError(f"Could not read '{path.name}' as a project: {exc}") from exc
    if not isinstance(document, dict) or document.get("format") != FORMAT:
        raise ProjectError(f"'{path.name}' is not a wrong-note MIDI cleaner project.")
    if document.get("version") != VERSION:
        raise ProjectError(f"'{path.name}' was saved by a different version of this tool "
                           f"(project format {document.get('version')}, this one reads {VERSION}).")
    edits = document.get("edits") or {}
    project_dir = path.resolve().parent
    perfect_path, perfect_hash = _locate(document.get("perfect"), project_dir, "perfect source")
    human_path, human_hash = _locate(document.get("human"), project_dir, "human MIDI")
    return Project(perfect_path, human_path, perfect_hash, human_hash,
                   _pairs(edits.get("remove", []), "removed notes"), _pairs(edits.get("add", []), "added notes"))


def restore_state(project: Project, session: EditSession) -> tuple[EditState, int]:
    """The saved edits as a state for `session`, and how many could not be matched to a proposal.

    An edit is skipped when its note no longer exists or the cleaner no longer proposes it, which
    only happens if the files or the cleaning rules changed since the project was saved.
    """
    perfect, human, plan = session.perfect, session.human, session.plan
    human_at = {(int(t), int(e)): h for h, (t, e) in enumerate(zip(human.tracks, human.on_events))}
    perfect_at = {(int(t), int(e)): p for p, (t, e) in enumerate(zip(perfect.tracks, perfect.on_events))}
    proposed_removals = {int(h) for h in plan.remove}
    add_of_source = {int(p): k for k, p in enumerate(plan.add_source)}

    removed = {human_at[key] for key in project.removed if human_at.get(key) in proposed_removals}
    added = {add_of_source[perfect_at[key]] for key in project.added
             if perfect_at.get(key) in add_of_source}
    skipped = len(project.removed) + len(project.added) - len(removed) - len(added)
    return EditState(frozenset(removed), frozenset(added)), skipped
