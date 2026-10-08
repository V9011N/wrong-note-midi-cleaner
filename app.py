"""Human MIDI Cleaner.

Pick a perfect source and a human MIDI; the match confidence appears automatically. Then choose
which kinds of fix may be applied (add missing notes / remove extra notes) and click Clean.
"""

from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from midi_cleaner.cleaning import CleaningPlan, build_cleaned_midi, plan_cleaning
from midi_cleaner.confidence import LIKELY_MATCH, UNCERTAIN_MATCH, MatchResult, analyze
from midi_cleaner.loader import MidiData, MidiLoadError, load_midi

MIDI_TYPES = [("MIDI files", "*.mid *.midi *.MID *.MIDI"), ("All files", "*.*")]
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
        # (perfect, human, result, plan) of the finished analysis the Clean button works from
        self._analysis: tuple[MidiData, MidiData, MatchResult, CleaningPlan] | None = None

        self._build_inputs()
        self._build_result()
        self._build_cleaning()
        self._refresh_cleaning()
        for var in (self.perfect_var, self.human_var):
            var.trace_add("write", lambda *_: self._schedule_analysis())
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
        self.clean_button = ttk.Button(frame, text="Clean", command=self._clean)
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
            self._results.put((run_id, perfect, human, result, plan, None))
        except MidiLoadError as exc:
            self._results.put((run_id, None, None, None, None, str(exc)))
        except Exception as exc:  # keep the UI usable whatever the analysis hits
            self._results.put((run_id, None, None, None, None, f"Unexpected error: {exc!r}"))

    def _poll_results(self) -> None:
        try:
            while True:
                run_id, perfect, human, result, plan, error = self._results.get_nowait()
                if run_id != self._run_id:
                    continue
                self.progress.stop()
                self.progress.grid_remove()
                if error:
                    self._show_idle(error, color=BAD)
                else:
                    self._show_result(perfect, human, result)
                    self._offer_cleaning(perfect, human, result, plan)
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
        """Sync labels, checkbox availability and the Clean button with the current state."""
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
            self.hint.configure(text="Select at least one option to enable Clean.", foreground=BAD)
        else:
            self.clean_button.state(["!disabled"])
            self.hint.configure(text="The options above were selected automatically from what was found. "
                                     "You may change the selection, but at least one must be chosen.",
                                foreground="#555555")

    def _ask_save_path(self, human: MidiData) -> str:
        """Open the system file browser for the cleaned file's destination ('' if cancelled)."""
        cleaned_dir = human.path.parent / "Cleaned"
        return filedialog.asksaveasfilename(
            title="Save cleaned MIDI", defaultextension=".mid", filetypes=MIDI_TYPES,
            initialdir=cleaned_dir if cleaned_dir.is_dir() else human.path.parent,
            initialfile=f"{human.path.stem} - CLEANED.mid")

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
        while True:
            chosen = self._ask_save_path(human)
            if not chosen:
                self.clean_status.configure(text="Cleaning finished, but nothing was saved.", foreground=MIDDLE)
                return
            target = Path(chosen)
            if target.exists() and target.resolve() == human.path.resolve():
                messagebox.showwarning("Original file", "The original human MIDI is never overwritten. "
                                                        "Please choose a different file name or folder.")
                continue
            break
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
