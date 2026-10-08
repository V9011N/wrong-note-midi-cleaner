"""File dialogs shared by the main window and the editor."""

from __future__ import annotations

from pathlib import Path
from tkinter import Misc, filedialog, messagebox

from .project import PROJECT_EXTENSION

MIDI_TYPES = [("MIDI files", "*.mid *.midi *.MID *.MIDI"), ("All files", "*.*")]
PROJECT_TYPES = [("Cleaner projects", f"*{PROJECT_EXTENSION}"), ("All files", "*.*")]


def ask_midi_save_path(parent: Misc, human_path: Path, title: str = "Save cleaned MIDI") -> Path | None:
    """Where to write a cleaned copy of the human MIDI; None if cancelled.

    The original file is never offered as a destination: choosing it asks again.
    """
    cleaned_dir = human_path.parent / "Cleaned"
    while True:
        chosen = filedialog.asksaveasfilename(
            parent=parent, title=title, defaultextension=".mid", filetypes=MIDI_TYPES,
            initialdir=cleaned_dir if cleaned_dir.is_dir() else human_path.parent,
            initialfile=f"{human_path.stem} - CLEANED.mid")
        if not chosen:
            return None
        target = Path(chosen)
        if target.exists() and target.resolve() == human_path.resolve():
            messagebox.showwarning("Original file", "The original human MIDI is never overwritten. "
                                                    "Please choose a different file name or folder.", parent=parent)
            continue
        return target


def ask_project_save_path(parent: Misc, human_path: Path) -> Path | None:
    chosen = filedialog.asksaveasfilename(
        parent=parent, title="Save project", defaultextension=PROJECT_EXTENSION, filetypes=PROJECT_TYPES,
        initialdir=human_path.parent, initialfile=f"{human_path.stem}{PROJECT_EXTENSION}")
    return Path(chosen) if chosen else None


def ask_project_open_path(parent: Misc, start_dir: Path) -> Path | None:
    chosen = filedialog.askopenfilename(parent=parent, title="Open project", filetypes=PROJECT_TYPES,
                                        initialdir=start_dir if start_dir.is_dir() else None)
    return Path(chosen) if chosen else None
