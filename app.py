"""Human MIDI Cleaner.

Pick a perfect source and a human MIDI; the match confidence appears automatically. Then choose
which kinds of fix may be applied (add missing notes / remove extra notes) and click Full Clean,
or open the editor to see how the two files' notes pair up and clean only the parts you pick.
A project saved from the editor can be reopened with "Open Project".
"""

from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from midi_cleaner import dialogs
from midi_cleaner.cleaning import CleaningPlan, build_cleaned_midi, plan_cleaning
from midi_cleaner.comparison import Comparison, build_comparison
from midi_cleaner.confidence import LIKELY_MATCH, UNCERTAIN_MATCH, MatchResult, analyze
from midi_cleaner.dialogs import MIDI_TYPES
from midi_cleaner.editing import EditSession
from midi_cleaner.editor import EditorWindow
from midi_cleaner.loader import MidiData, MidiLoadError, load_midi
from midi_cleaner.project import Project, ProjectError, load_project, restore_state
DEBOUNCE_MS = 400
GOOD, MIDDLE, BAD = "#1a7f37", "#b35900", "#c62828"


def clean_path(raw: str) -> str:
    return raw.strip().strip('"').strip("'")


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Human MIDI Cleaner")
        self.geometry("860x800")
        self.minsize(760, 720)
        self.columnconfigure(0, weight=1)

        self.perfect_var = tk.StringVar()
        self.human_var = tk.StringVar()
        self._debounce_id: str | None = None
        self._run_id = 0  # bumped on every new analysis so stale results are dropped
        self._results: queue.Queue = queue.Queue()
        self._start_dir = Path(__file__).parent / "MIDIs"
        self.add_var = tk.BooleanVar(value=False)
        self.remove_var = tk.BooleanVar(value=False)
        # (perfect, human, result, plan) of the finished analysis the Full Clean button works from
        self._analysis: tuple[MidiData, MidiData, MatchResult, CleaningPlan] | None = None
        # (perfect, human, plan, comparison) of the same analysis, which the editor works from
        self._context: tuple[MidiData, MidiData, CleaningPlan, Comparison] | None = None
        self._editor: EditorWindow | None = None  # the editor open on the current analysis
        self._editors: list[EditorWindow] = []  # every editor still open; they outlive a change of files
        self._pending_project: tuple[Project, Path] | None = None  # to open once its files are analysed

        self._build_inputs()
        self._build_result()
        self._build_cleaning()
        self._refresh_cleaning()
        for var in (self.perfect_var, self.human_var):
            var.trace_add("write", lambda *_: self._schedule_analysis())
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_results)

    # ---- layout ---------------------------------------------------------------

    def _build_inputs(self) -> None:
        frame = ttk.LabelFrame(self, text="MIDI files", padding=10)
        frame.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        frame.columnconfigure(1, weight=1)
        for row, (label, var) in enumerate((("Perfect source:", self.perfect_var),
                                            ("Human MIDI:", self.human_var))):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(frame, textvariable=var).grid(row=row, column=1, sticky="ew", padx=8)
            ttk.Button(frame, text="Browse...", command=lambda v=var, t=label: self._browse(v, t)
                       ).grid(row=row, column=2)
        ttk.Button(frame, text="Open Project...", command=self._open_project).grid(row=2, column=2, pady=(6, 0))

    def _build_result(self) -> None:
        frame = ttk.LabelFrame(self, text="Match confidence", padding=10)
        frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=6)
        frame.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self.percent = ttk.Label(frame, text="--", font=("Segoe UI", 40, "bold"), anchor="center")
        self.percent.grid(row=0, column=0, sticky="ew")
        self.verdict = ttk.Label(frame, text="Choose both files to compare them.",
                                 font=("Segoe UI", 12), anchor="center")
        self.verdict.grid(row=1, column=0, sticky="ew")
        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.grid(row=2, column=0, sticky="ew", pady=8)
        self.progress.grid_remove()

        self.details = tk.Text(frame, height=13, wrap="word", relief="flat", state="disabled",
                               font=("Consolas", 10), background=self.cget("background"))
        self.details.grid(row=3, column=0, sticky="nsew")
        frame.rowconfigure(3, weight=1)
        self.compare_button = ttk.Button(frame, text="Open Editor...", command=self._open_editor,
                                         state="disabled")
        self.compare_button.grid(row=4, column=0, sticky="e", pady=(8, 0))

    def _build_cleaning(self) -> None:
        frame = ttk.LabelFrame(self, text="Cleaning", padding=10)
        frame.grid(row=2, column=0, sticky="ew", padx=12, pady=(6, 12))
        frame.columnconfigure(0, weight=1)
        self.findings = ttk.Label(frame, text="", wraplength=780, justify="left")
        self.findings.grid(row=0, column=0, columnspan=3, sticky="w")

        self.add_check = ttk.Checkbutton(frame, text="Add notes", variable=self.add_var,
                                         command=self._refresh_cleaning)
        self.remove_check = ttk.Checkbutton(frame, text="Remove notes", variable=self.remove_var,
                                            command=self._refresh_cleaning)
        self.add_check.grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.remove_check.grid(row=2, column=0, sticky="w")
        self.clean_button = ttk.Button(frame, text="Full Clean", command=self._clean)
        self.clean_button.grid(row=1, column=2, rowspan=2, sticky="e", ipadx=18, ipady=4)

        self.hint = ttk.Label(frame, text="", wraplength=780, justify="left", foreground="#555555")
        self.hint.grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.clean_status = ttk.Label(frame, text="", wraplength=780, justify="left")
        self.clean_status.grid(row=4, column=0, columnspan=3, sticky="w", pady=(4, 0))

    # ---- input handling -------------------------------------------------------

    def _browse(self, var: tk.StringVar, title: str) -> None:
        current = Path(clean_path(var.get()))
        initial = current.parent if current.parent.is_dir() else self._start_dir
        chosen = filedialog.askopenfilename(
            title=f"Select {title.rstrip(':').lower()}", filetypes=MIDI_TYPES,
            initialdir=initial if initial.is_dir() else None)
        if chosen:
            var.set(str(Path(chosen)))
            self._start_dir = Path(chosen).parent

    def _schedule_analysis(self) -> None:
        if self._debounce_id is not None:
            self.after_cancel(self._debounce_id)
        self._debounce_id = self.after(DEBOUNCE_MS, self._start_analysis)

    def _start_analysis(self) -> None:
        self._debounce_id = None
        self._run_id += 1  # invalidates any analysis still running
        self._analysis = None
        self._set_context(None)
        self.clean_status.configure(text="")
        self._refresh_cleaning()
        perfect, human = clean_path(self.perfect_var.get()), clean_path(self.human_var.get())
        if not perfect or not human:
            self._show_idle("Choose both files to compare them.")
            return
        for label, path in (("Perfect source", perfect), ("Human MIDI", human)):
            if not Path(path).is_file():
                self._show_idle(f"{label}: file not found.")
                return

        self.percent.configure(text="...", foreground="")
        self.verdict.configure(text="Analyzing...", foreground="")
        self._set_details("")
        self.progress.grid()
        self.progress.start(12)
        threading.Thread(target=self._worker, args=(self._run_id, perfect, human), daemon=True).start()

    def _worker(self, run_id: int, perfect_path: str, human_path: str) -> None:
        try:
            perfect, human = load_midi(perfect_path), load_midi(human_path)
            result, alignment = analyze(perfect, human)
            plan = plan_cleaning(perfect, human, alignment)
            comparison = build_comparison(perfect, human, alignment, plan)
            self._results.put((run_id, perfect, human, result, plan, comparison, None))
        except MidiLoadError as exc:
            self._results.put((run_id, None, None, None, None, None, str(exc)))
        except Exception as exc:  # keep the UI usable whatever the analysis hits
            self._results.put((run_id, None, None, None, None, None, f"Unexpected error: {exc!r}"))

    def _poll_results(self) -> None:
        try:
            while True:
                run_id, perfect, human, result, plan, comparison, error = self._results.get_nowait()
                if run_id != self._run_id:
                    continue
                self.progress.stop()
                self.progress.grid_remove()
                if error:
                    self._show_idle(error, color=BAD)
                else:
                    self._show_result(perfect, human, result)
                    self._offer_cleaning(perfect, human, result, plan)
                    self._set_context((perfect, human, plan, comparison))
                    self._open_pending_project()
        except queue.Empty:
            pass
        self.after(100, self._poll_results)

    # ---- display --------------------------------------------------------------

    def _show_idle(self, message: str, color: str = "") -> None:
        self.progress.stop()
        self.progress.grid_remove()
        self.percent.configure(text="--", foreground="")
        self.verdict.configure(text=message, foreground=color)
        self._set_details("")
        self._analysis = None
        self._pending_project = None  # its files can't be used
        self._set_context(None)
        self._refresh_cleaning()

    def _show_result(self, perfect: MidiData, human: MidiData, r: MatchResult) -> None:
        color = GOOD if r.confidence >= LIKELY_MATCH else MIDDLE if r.confidence >= UNCERTAIN_MATCH else BAD
        self.percent.configure(text=f"{r.confidence:.0%}", foreground=color)
        self.verdict.configure(text=r.verdict, foreground=color)

        pct = lambda x: f"{x:.0%}"  # noqa: E731
        lines = [
            f"{'':26}{'Perfect':>12}{'Human':>12}",
            f"{'Notes':26}{perfect.note_count:>12,}{human.note_count:>12,}",
            f"{'Length (s)':26}{perfect.span:>12.1f}{human.span:>12.1f}",
            "Structure",
            f"{'Coverage of the source':26}{pct(r.coverage):>12}   share of the piece the human plays",
            f"{'Aligned stretches':26}{r.segments:>12}"
            + ("   (more than 1: the human skipped or replayed material)" if r.segments > 1 else ""),
            "Evidence (what the confidence is built from)",
            f"{'Alignment similarity':26}{pct(r.alignment_similarity):>12}   pitch content along the tempo-warp",
            f"{'Perfect notes found':26}{pct(r.raw_recall):>12}"
            f"   in the human file (+/-{r.match_tolerance * 1000:.0f} ms)",
            f"{'Human notes found':26}{pct(r.raw_precision):>12}   in the perfect file",
            f"{'Chance level':26}{pct(r.chance_recall):>12}   expected from an unrelated alignment",
            f"{'  -> beyond chance':26}{pct(r.perfect_recall):>12}{pct(r.human_precision):>12}   (perfect / human)",
        ]
        for name, data in (("Perfect source", perfect), ("Human MIDI", human)):
            for warning in data.warnings:
                lines += ["", f"Note ({name}): {warning}"]
        self._set_details("\n".join(lines))

    # ---- editor and projects --------------------------------------------------

    def _set_context(self, context: tuple[MidiData, MidiData, CleaningPlan, Comparison] | None) -> None:
        """The analysis the editor would open on. Editors already open keep their own copy."""
        self._context = context
        self._editor = None
        self.compare_button.state(["!disabled" if context is not None else "disabled"])

    def _open_editor(self, project: Project | None = None, project_path: Path | None = None) -> None:
        if self._context is None:
            return
        if project is None and self._editor is not None and self._editor.winfo_exists():
            self._editor.lift()
            self._editor.focus_set()
            return
        perfect, human, plan, comparison = self._context
        session = EditSession(perfect, human, plan, comparison)
        notice = ""
        if project is not None:
            state, skipped = restore_state(project, session)
            session.load_state(state)
            notice = f"Opened project {project_path.name}: {state.count:,} edit(s) restored."
            if skipped:
                notice += (f" {skipped:,} saved edit(s) could not be matched to a note the cleaner proposes "
                           "(the files or the cleaning rules changed), so they were skipped.")
        self._editors = [e for e in self._editors if e.winfo_exists()]
        self._editor = EditorWindow(self, session, f"{perfect.path.name}  vs  {human.path.name}",
                                    project_path=project_path, notice=notice)
        self._editors.append(self._editor)

    def _open_project(self) -> None:
        path = dialogs.ask_project_open_path(self, self._start_dir)
        if path is None:
            return
        try:
            project = load_project(path)
        except ProjectError as exc:
            messagebox.showerror("Open project", str(exc))
            return
        missing = project.missing_files()
        if missing:
            messagebox.showerror("Open project", "These files from the project could not be found:\n\n"
                                 + "\n".join(str(m) for m in missing))
            return
        changed = project.changed_files()
        if changed and not messagebox.askyesno(
                "Files changed", "These files were modified since the project was saved: "
                f"{', '.join(changed)}.\n\nThe saved edits may no longer line up. Open it anyway?"):
            return
        self._pending_project = (project, path)  # the editor opens once the two files are analysed
        self.perfect_var.set(str(project.perfect_path))
        self.human_var.set(str(project.human_path))

    def _open_pending_project(self) -> None:
        if self._pending_project is None or self._context is None:
            return
        (project, path), self._pending_project = self._pending_project, None
        perfect, human = self._context[:2]
        if (perfect.path.resolve(), human.path.resolve()) == (project.perfect_path.resolve(),
                                                              project.human_path.resolve()):
            self._open_editor(project, path)

    def _on_close(self) -> None:
        """Quit, unless an editor still holds edits the user wants to save first."""
        for editor in [e for e in self._editors if e.winfo_exists()]:
            if not editor.close():
                return
        self.destroy()

    # ---- cleaning -------------------------------------------------------------

    def _offer_cleaning(self, perfect: MidiData, human: MidiData, result: MatchResult,
                        plan: CleaningPlan) -> None:
        """Tick the boxes for whatever the algorithm found needs doing, then let the user adjust."""
        self._analysis = (perfect, human, result, plan)
        matches = result.confidence >= UNCERTAIN_MATCH
        self.add_var.set(matches and plan.n_add > 0)
        self.remove_var.set(matches and plan.n_remove > 0)
        self.clean_status.configure(text="")
        self._refresh_cleaning()

    def _refresh_cleaning(self) -> None:
        """Sync labels, checkbox availability and the Full Clean button with the current state."""
        if self._analysis is None:
            for widget in (self.add_check, self.remove_check, self.clean_button):
                widget.state(["disabled"])
            self.add_check.configure(text="Add notes")
            self.remove_check.configure(text="Remove notes")
            self.findings.configure(text="Cleaning is available once both files have been compared.",
                                    foreground="")
            self.hint.configure(text="")
            return

        _, _, result, plan = self._analysis
        if result.confidence < UNCERTAIN_MATCH:
            self.add_check.configure(text="Add notes")
            self.remove_check.configure(text="Remove notes")
            for widget in (self.add_check, self.remove_check, self.clean_button):
                widget.state(["disabled"])
            self.findings.configure(text="Cleaning is disabled: these files don't look like the same piece, "
                                         "so editing one to match the other would damage it.", foreground=BAD)
            self.hint.configure(text="")
            return

        self.add_check.configure(text=f"Add notes  ({plan.n_add:,} missing from the human performance)")
        self.remove_check.configure(text=f"Remove notes  ({plan.n_remove:,} not in the perfect source)")
        for widget in (self.add_check, self.remove_check):
            widget.state(["!disabled"])
        found = f"Found {plan.n_remove:,} extra note(s) to remove and {plan.n_add:,} missing note(s) to add."
        left_alone = plan.displaced + plan.untrusted
        if left_alone:
            found += (f" {left_alone:,} other difference(s) were left alone "
                      "(timing differences, or places where the alignment is uncertain).")
        if result.confidence < LIKELY_MATCH:
            found += " The match is uncertain, so check the result carefully."
        self.findings.configure(text=found, foreground="")
        if not (self.add_var.get() or self.remove_var.get()):
            self.clean_button.state(["disabled"])
            self.hint.configure(text="Select at least one option to enable Full Clean.", foreground=BAD)
        else:
            self.clean_button.state(["!disabled"])
            self.hint.configure(text="The options above were selected automatically from what was found. "
                                     "You may change the selection, but at least one must be chosen.",
                                foreground="#555555")

    def _clean(self) -> None:
        if self._analysis is None or not (self.add_var.get() or self.remove_var.get()):
            return
        _, human, _, plan = self._analysis
        add, remove = self.add_var.get(), self.remove_var.get()
        try:
            cleaned = build_cleaned_midi(plan, human, remove=remove, add=add)
        except Exception as exc:
            messagebox.showerror("Cleaning failed", f"Could not clean the MIDI:\n{exc}")
            return

        # The browser opens only now, after the cleaning is done.
        target = dialogs.ask_midi_save_path(self, human.path)
        if target is None:
            self.clean_status.configure(text="Cleaning finished, but nothing was saved.", foreground=MIDDLE)
            return
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            cleaned.save(str(target))
        except OSError as exc:
            messagebox.showerror("Save failed", f"Could not save the file:\n{exc}")
            return
        done = []
        if remove:
            done.append(f"removed {plan.n_remove:,} note(s)")
        if add:
            done.append(f"added {plan.n_add:,} note(s)")
        self.clean_status.configure(text=f"Saved {target}  ({', '.join(done)}).", foreground=GOOD)

    def _set_details(self, text: str) -> None:
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", text)
        self.details.configure(state="disabled")


def enable_high_dpi() -> None:
    """Keep text crisp on scaled Windows displays."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass


if __name__ == "__main__":
    enable_high_dpi()
    App().mainloop()
