"""The editor: the perfect source's piano roll above the human performance's, to clean by hand.

Human notes are green when they match the score, red when the score doesn't have them, blue
when they are notes the cleaner would add, and grey when it could not tell. Click a note, or
drag a box around several, to draw a line to each note's counterpart in the other roll. Each
roll has its own filters, and both rolls share one zoom and scroll position.

Nothing is changed until "Clean Selection": it applies the cleaner's proposals (delete the wrong
notes, insert the missing ones) to the selected notes only, and leaves everything else alone.
Every such step can be undone and redone. The edits can be kept as a project, or written out as a
cleaned MIDI; the original files are never modified.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np

from . import dialogs
from .cleaning import ADD, DISPLACED, MATCHED, REMOVE
from .comparison import ADDED, REMOVED, Comparison, Roll, describe_note, note_name
from .editing import EditSession
from .project import save_project

PERFECT_COLOR = "#555b66"  # neutral, so that blue stays "to be added"
CORRECT_COLOR = "#2e9e4f"
WRONG_COLOR = "#d62f2f"
ADDED_COLOR = "#1f6fe0"
UNJUDGED_COLOR = "#a8a8a8"
REMOVED_FILL = "#f3c9c9"  # a deleted note stays on the roll as a pale, dashed ghost
LINK_COLOR = "#111111"
PANEL_BG, BLACK_KEY_BG = "#fbfbfc", "#eef0f4"
GRID_COLOR, OCTAVE_COLOR = "#dfe2e8", "#b4bac6"

GUTTER = 52  # left strip with the key names
RULER = 24
TITLE = 18
GAP = 46  # between the rolls: room for the human roll's title and the lines
MARGIN = 10
MIN_PPS, MAX_PPS = 6.0, 500.0  # pixels per second
MIN_ROWS = 8  # fewest pitches the vertical zoom shows
MAX_ROW_H = 36.0
LABEL_ALL_ROWS_H = 12.0  # rows at least this tall name every pitch, not just the Cs
DRAG_THRESHOLD = 4
HIT_RADIUS = 2
TICK_STEPS = (0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800)
SHIFT, CONTROL = 0x1, 0x4
HINT = ("Click a note to see its match, or drag a box (Ctrl+A for everything shown) to select a group, "
        "then Clean Selection. Ctrl+Z undo, Ctrl+Y redo, Ctrl+S save project, Ctrl+E export. "
        "Scroll pans in time, Shift+scroll in pitch; Ctrl+scroll zooms in time, Ctrl+Shift+scroll in pitch; "
        "right-drag pans; Esc clears.")
CORRECT = (MATCHED, DISPLACED, ADDED)  # human statuses the "correct notes" filter covers
WRONG = (REMOVE, REMOVED)  # ... the "wrong notes" filter
JUDGED = (MATCHED, DISPLACED, ADDED, REMOVE, REMOVED, ADD)  # everything else is "not judged"
# status -> (fill, outline, dash)
HUMAN_STYLE = {
    MATCHED: (CORRECT_COLOR, "", None), DISPLACED: (CORRECT_COLOR, "", None),
    ADDED: (CORRECT_COLOR, ADDED_COLOR, None),  # a note that was just inserted keeps a blue edge
    REMOVE: (WRONG_COLOR, "", None), REMOVED: (REMOVED_FILL, WRONG_COLOR, (3, 2)),
    ADD: (ADDED_COLOR, "", None),
}
UNJUDGED_STYLE = (UNJUDGED_COLOR, "", None)


class EditorWindow(tk.Toplevel):
    def __init__(self, master: tk.Misc, session: EditSession, title: str,
                 project_path: Path | None = None, notice: str = "") -> None:
        super().__init__(master)
        self._title = title
        self.geometry("1240x860")
        self.minsize(1000, 520)
        self.session = session
        self.comp = comparison = session.view()  # the comparison with the applied edits reflected
        self.project_path = project_path
        self._notice = notice
        self._pps = 40.0  # pixels per second
        self._t0 = comparison.t_min  # time at the left edge of the rolls
        self._vis_rows = self._rows_total()  # pitches shown from top to bottom of a roll
        self._p_top = comparison.pitch_hi  # the highest pitch shown
        self._row_h = 6.0
        self._top = {"p": 0.0, "h": 0.0}  # canvas y of each roll's top edge
        self._sel: dict[str, set[int]] = {"p": set(), "h": set()}
        self._items: dict[int, tuple[str, int]] = {}  # canvas item -> (roll, note index)
        self._press: dict | None = None
        self._pan: tuple[float, float, float, int] | None = None
        self._ready = False
        self._only_missing = tk.BooleanVar(self, False)
        self._show_correct = tk.BooleanVar(self, True)
        self._show_wrong = tk.BooleanVar(self, True)
        self._show_added = tk.BooleanVar(self, False)
        self._show_unjudged = tk.BooleanVar(self, True)
        self._shown: dict[str, np.ndarray] = {}  # per roll: which notes the filters let through
        self._recompute_shown()

        self._build()
        self._update_actions()
        self._update_title()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.canvas.focus_set()
        if notice:
            self.info.configure(text=notice)

    # ---- widgets --------------------------------------------------------------

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        filters = ttk.Frame(self, padding=(10, 8, 10, 0))
        filters.grid(row=0, column=0, columnspan=2, sticky="ew")
        perfect_box = ttk.LabelFrame(filters, text="Perfect source (top)", padding=(8, 2, 8, 4))
        perfect_box.pack(side="left", padx=(0, 10))
        self._filter(perfect_box, PERFECT_COLOR, "Show only notes missing in target MIDI", self._only_missing)
        human_box = ttk.LabelFrame(filters, text="Human performance (bottom)", padding=(8, 2, 8, 4))
        human_box.pack(side="left")
        for color, text, var in ((CORRECT_COLOR, "Show correct notes", self._show_correct),
                                 (WRONG_COLOR, "Show wrong notes", self._show_wrong),
                                 (ADDED_COLOR, "Show notes to be added", self._show_added),
                                 (UNJUDGED_COLOR, "Show not-judged notes", self._show_unjudged)):
            self._filter(human_box, color, text, var)

        bar = ttk.Frame(self, padding=(10, 6, 10, 2))
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        ttk.Label(bar, text="Zoom time").pack(side="left")
        for text, factor in (("\u2212", 1 / 1.5), ("+", 1.5)):
            ttk.Button(bar, text=text, width=3, command=lambda f=factor: self._zoom(f)).pack(side="left", padx=2)
        ttk.Label(bar, text="  Zoom pitch").pack(side="left")
        for text, factor in (("\u2212", 1 / 1.5), ("+", 1.5)):
            ttk.Button(bar, text=text, width=3, command=lambda f=factor: self._zoom_pitch(f)).pack(side="left", padx=2)
        ttk.Button(bar, text="Fit", width=5, command=self._fit).pack(side="left", padx=(12, 0))

        # Pack right to left: Export, Save, [Clean Selection], Redo, Undo.
        self.export_button = ttk.Button(bar, text="Export cleaned MIDI", command=self.export_midi)
        self.export_button.pack(side="right", padx=(4, 0))
        self.save_button = ttk.Button(bar, text="Save Project", command=self.save)
        self.save_button.pack(side="right", padx=(12, 0))
        self.redo_button = ttk.Button(bar, text="Redo", width=6, command=self.redo)
        self.redo_button.pack(side="right", padx=(4, 0))
        self.undo_button = ttk.Button(bar, text="Undo", width=6, command=self.undo)
        self.undo_button.pack(side="right", padx=(4, 0))
        self.clean_button = ttk.Button(bar, text="Clean Selection", command=self.clean_selection)
        self._clean_before = self.redo_button  # where the Clean Selection button goes when it appears

        self.canvas = tk.Canvas(self, background="white", highlightthickness=0)
        self.canvas.grid(row=2, column=0, sticky="nsew", padx=(10, 0))
        self.vscroll = ttk.Scrollbar(self, orient="vertical", command=self._vscroll_command)
        self.vscroll.grid(row=2, column=1, sticky="ns", padx=(0, 10))
        self.scroll = ttk.Scrollbar(self, orient="horizontal", command=self._scroll_command)
        self.scroll.grid(row=3, column=0, sticky="ew", padx=(10, 0))
        self.info = ttk.Label(self, text=HINT, wraplength=1100, justify="left", padding=(10, 6))
        self.info.grid(row=4, column=0, columnspan=2, sticky="ew")

        c = self.canvas
        c.bind("<Configure>", self._on_resize)
        c.bind("<ButtonPress-1>", self._on_press)
        c.bind("<B1-Motion>", self._on_drag)
        c.bind("<ButtonRelease-1>", self._on_release)
        for button in (2, 3):  # middle and right (the right one is button 2 on macOS)
            c.bind(f"<ButtonPress-{button}>", self._on_pan_start)
            c.bind(f"<B{button}-Motion>", self._on_pan_move)
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            c.bind(sequence, self._on_wheel)
        self.bind("<Escape>", lambda _e: self._set_selection({"p": set(), "h": set()}))
        for sequences, command in ((("<Control-z>", "<Control-Z>"), self.undo),
                                   (("<Control-y>", "<Control-Y>", "<Control-Shift-Z>", "<Control-Shift-z>"), self.redo),
                                   (("<Control-s>", "<Control-S>"), self.save),
                                   (("<Control-e>", "<Control-E>"), self.export_midi),
                                   (("<Control-a>", "<Control-A>"), self.select_all)):
            for sequence in sequences:
                self.bind(sequence, lambda _e, run=command: (run(), "break")[1])
        self.info.bind("<Configure>", lambda e: self.info.configure(wraplength=max(e.width - 20, 200)))

    def _filter(self, parent: tk.Misc, color: str, text: str, var: tk.BooleanVar) -> None:
        """A checkbox led by a swatch of the colour it controls."""
        tk.Label(parent, width=2, bg=color).pack(side="left", padx=(0, 3))
        ttk.Checkbutton(parent, text=text, variable=var, command=self._on_filters_changed
                        ).pack(side="left", padx=(0, 12))

    # ---- geometry -------------------------------------------------------------

    def _x(self, t):
        return GUTTER + (t - self._t0) * self._pps

    def _t(self, x):
        return self._t0 + (x - GUTTER) / self._pps

    def _y(self, side: str, pitch):
        """Top edge of a pitch's row (the pitch at the top of the roll is row 0)."""
        return self._top[side] + (self._p_top - pitch) * self._row_h

    def _bottom(self, side: str) -> float:
        return self._top[side] + self._vis_rows * self._row_h

    def _rows_total(self) -> int:
        return self.comp.pitch_hi - self.comp.pitch_lo + 1

    def _span(self) -> float:
        """Seconds of music visible across the rolls."""
        return max(self.canvas.winfo_width() - GUTTER, 1) / self._pps

    def _bounds(self) -> tuple[float, float]:
        return self.comp.t_min - 1.0, self.comp.t_max + 1.0

    def _clamp_view(self) -> None:
        lo, hi = self._bounds()
        self._t0 = min(max(self._t0, lo), max(hi - self._span(), lo))
        total = self._rows_total()
        self._vis_rows = int(min(max(self._vis_rows, min(MIN_ROWS, total)), total))
        self._p_top = int(min(max(self._p_top, self.comp.pitch_lo + self._vis_rows - 1), self.comp.pitch_hi))

    def _layout(self) -> None:
        free = self.canvas.winfo_height() - RULER - TITLE - GAP - MARGIN
        self._row_h = float(min(max(free / (2 * self._vis_rows), 1.0), MAX_ROW_H))
        self._top["p"] = RULER + TITLE
        self._top["h"] = self._top["p"] + self._vis_rows * self._row_h + GAP

    def _refresh(self) -> None:
        self._clamp_view()
        self._layout()
        self._redraw()

    def _on_resize(self, _event) -> None:
        if not self._ready:
            self._ready = True
            self._fit()
            return
        self._refresh()

    # ---- view changes ---------------------------------------------------------

    def _fit(self) -> None:
        """Show every pitch, and the whole piece if it is short enough to read, else its start."""
        lo, hi = self._bounds()
        width = max(self.canvas.winfo_width() - GUTTER, 1)
        self._pps = float(np.clip(width / (hi - lo), MIN_PPS, MAX_PPS))
        self._t0 = lo
        self._vis_rows, self._p_top = self._rows_total(), self.comp.pitch_hi
        self._refresh()

    def _zoom(self, factor: float, anchor_x: float | None = None) -> None:
        """Zoom the time axis about a canvas x (default: the middle)."""
        if anchor_x is None:
            anchor_x = GUTTER + (self.canvas.winfo_width() - GUTTER) / 2
        anchor_t = self._t(anchor_x)
        self._pps = float(np.clip(self._pps * factor, MIN_PPS, MAX_PPS))
        self._t0 = anchor_t - (anchor_x - GUTTER) / self._pps
        self._refresh()

    def _zoom_pitch(self, factor: float, anchor_y: float | None = None) -> None:
        """Zoom the pitch axis (both rolls together) about a canvas y (default: the middle)."""
        side = "h" if anchor_y is not None and anchor_y > (self._bottom("p") + self._top["h"]) / 2 else "p"
        fraction = 0.5 if anchor_y is None else float(np.clip(
            (anchor_y - self._top[side]) / (self._vis_rows * self._row_h), 0.0, 1.0))
        anchor_pitch = self._p_top - fraction * self._vis_rows
        rows = int(round(self._vis_rows / factor))
        if rows == self._vis_rows:  # always move at least one row
            rows += -1 if factor > 1 else 1
        self._vis_rows = min(max(rows, min(MIN_ROWS, self._rows_total())), self._rows_total())
        self._p_top = int(round(anchor_pitch + fraction * self._vis_rows))
        self._refresh()

    def _scroll_command(self, kind: str, *args: str) -> None:
        lo, hi = self._bounds()
        if kind == "moveto":
            self._t0 = lo + float(args[0]) * (hi - lo)
        elif kind == "scroll":
            step = self._span() * (0.9 if args[1] == "pages" else 0.1)
            self._t0 += int(args[0]) * step
        self._refresh()

    def _vscroll_command(self, kind: str, *args: str) -> None:
        if kind == "moveto":
            self._p_top = self.comp.pitch_hi - int(round(float(args[0]) * self._rows_total()))
        elif kind == "scroll":
            step = self._vis_rows - 1 if args[1] == "pages" else max(self._vis_rows // 10, 1)
            self._p_top -= int(args[0]) * step  # scrolling down reveals lower pitches
        self._refresh()

    def _on_wheel(self, event) -> None:
        direction = 1 if event.num == 4 or (event.num != 5 and event.delta > 0) else -1
        control, shift = bool(event.state & CONTROL), bool(event.state & SHIFT)
        if control and shift:
            self._zoom_pitch(1.25 ** direction, event.y)
        elif control:
            self._zoom(1.25 ** direction, event.x)
        elif shift:
            self._p_top += direction * max(self._vis_rows // 10, 1)
            self._refresh()
        else:
            self._t0 -= direction * self._span() * 0.1
            self._refresh()

    def _on_pan_start(self, event) -> None:
        self._pan = (event.x, event.y, self._t0, self._p_top)

    def _on_pan_move(self, event) -> None:
        if self._pan is not None:
            x0, y0, t0, p_top = self._pan
            self._t0 = t0 - (event.x - x0) / self._pps
            self._p_top = p_top + int(round((event.y - y0) / self._row_h))  # the roll follows the pointer
            self._refresh()

    # ---- filters --------------------------------------------------------------

    def _recompute_shown(self) -> None:
        """Which notes of each roll the filter checkboxes currently let through."""
        p_status, h_status = self.comp.perfect.status, self.comp.human.status
        self._shown["p"] = (~np.isin(p_status, CORRECT) if self._only_missing.get()  # inserted = no longer missing
                            else np.ones(len(p_status), bool))
        self._shown["h"] = ((np.isin(h_status, CORRECT) & self._show_correct.get())
                            | (np.isin(h_status, WRONG) & self._show_wrong.get())
                            | ((h_status == ADD) & self._show_added.get())
                            | (~np.isin(h_status, JUDGED) & self._show_unjudged.get()))

    def _on_filters_changed(self) -> None:
        self._recompute_shown()
        # a note that disappears is no longer selected, so its line goes with it
        self._set_selection({side: {i for i in ids if self._shown[side][i]} for side, ids in self._sel.items()})

    # ---- drawing --------------------------------------------------------------

    def _redraw(self) -> None:
        c = self.canvas
        width = c.winfo_width()
        c.delete("all")
        self._items.clear()
        if width <= GUTTER + 10:
            return
        t_a, t_b = self._t(GUTTER), self._t(width)
        for side, name in (("p", "Perfect source, laid onto the performance's timing"),
                           ("h", "Human performance")):
            self._draw_panel(side, name, width)
        self._draw_time_grid(width, t_a, t_b)
        highlight = self._highlighted()
        self._draw_notes("p", t_a, t_b, highlight)
        self._draw_notes("h", t_a, t_b, highlight)
        self._draw_links(width)
        self._update_scrollbars()

    def _visible_pitches(self) -> range:
        return range(self._p_top - self._vis_rows + 1, self._p_top + 1)

    def _draw_panel(self, side: str, title: str, width: int) -> None:
        c = self.canvas
        top, bottom = self._top[side], self._bottom(side)
        c.create_text(GUTTER, top - 4, text=title, anchor="sw", fill="#444444", font=("TkDefaultFont", 9, "bold"))
        c.create_rectangle(GUTTER, top, width, bottom, fill=PANEL_BG, outline=GRID_COLOR)
        name_every_row = self._row_h >= LABEL_ALL_ROWS_H
        for pitch in self._visible_pitches():
            y = self._y(side, pitch)
            if pitch % 12 in (1, 3, 6, 8, 10):
                c.create_rectangle(GUTTER, y, width, y + self._row_h, fill=BLACK_KEY_BG, outline="")
            if pitch % 12 == 0:
                c.create_line(GUTTER, y + self._row_h, width, y + self._row_h, fill=OCTAVE_COLOR)
            if pitch % 12 == 0 or name_every_row:
                c.create_text(GUTTER - 6, y + self._row_h / 2, text=note_name(pitch), anchor="e",
                              fill="#555555", font=("TkDefaultFont", 8))

    def _draw_time_grid(self, width: int, t_a: float, t_b: float) -> None:
        c = self.canvas
        step = next((s for s in TICK_STEPS if s * self._pps >= 70), TICK_STEPS[-1])
        first = int(np.ceil(max(t_a, 0.0) / step))
        for k in range(first, int(t_b / step) + 1):
            t = k * step
            x = self._x(t)
            for side in "ph":
                c.create_line(x, self._top[side], x, self._bottom(side), fill=GRID_COLOR)
            label = f"{int(t // 60)}:{t % 60:02.0f}" if step >= 1 else f"{t:.2f}s"
            c.create_line(x, RULER - 5, x, RULER, fill=OCTAVE_COLOR)
            c.create_text(x + 3, RULER - 6, text=label, anchor="sw", fill="#555555", font=("TkDefaultFont", 8))

    def _draw_notes(self, side: str, t_a: float, t_b: float, highlight: dict[str, set[int]]) -> None:
        roll = self._roll(side)
        low, high = self._p_top - self._vis_rows + 1, self._p_top
        visible = np.flatnonzero((roll.end >= t_a) & (roll.start <= t_b) & self._shown[side]
                                 & (roll.pitches >= low) & (roll.pitches <= high))
        x1 = self._x(roll.start[visible])
        x2 = np.maximum(self._x(roll.end[visible]), x1 + 2.0)
        y1 = self._y(side, roll.pitches[visible]) + (0.5 if self._row_h >= 4 else 0.0)
        height = self._row_h - (1.0 if self._row_h >= 4 else 0.0)
        marked = highlight[side]
        create, items = self.canvas.create_rectangle, self._items
        for n, i in enumerate(visible):
            i = int(i)
            fill, outline, dash = HUMAN_STYLE.get(int(roll.status[i]), UNJUDGED_STYLE) if side == "h" \
                else (PERFECT_COLOR, "", None)
            if i in marked:
                outline, dash = "black", None
            item = create(x1[n], y1[n], x2[n], y1[n] + height, fill=fill, outline=outline, width=1,
                          **({"dash": dash} if dash else {}))
            items[item] = (side, i)

    def _draw_links(self, width: int) -> None:
        comp = self.comp
        low, high = self._p_top - self._vis_rows + 1, self._p_top
        for p, h in sorted(self._pairs()):
            if not low <= comp.perfect.pitches[p] <= high:  # partners share a pitch, so both are off-screen
                continue
            xp = self._x((comp.perfect.start[p] + comp.perfect.end[p]) / 2)
            xh = self._x((comp.human.start[h] + comp.human.end[h]) / 2)
            if max(xp, xh) < GUTTER or min(xp, xh) > width:
                continue
            yp = self._y("p", comp.perfect.pitches[p]) + self._row_h / 2
            yh = self._y("h", comp.human.pitches[h]) + self._row_h / 2
            self.canvas.create_line(xp, yp, xh, yh, fill=LINK_COLOR, width=1.5)

    def _update_scrollbars(self) -> None:
        lo, hi = self._bounds()
        first = (self._t0 - lo) / (hi - lo)
        self.scroll.set(first, min(first + self._span() / (hi - lo), 1.0))
        total = self._rows_total()
        top = (self.comp.pitch_hi - self._p_top) / total
        self.vscroll.set(top, min(top + self._vis_rows / total, 1.0))

    # ---- selection ------------------------------------------------------------

    def _roll(self, side: str) -> Roll:
        return self.comp.human if side == "h" else self.comp.perfect

    def _pairs(self) -> set[tuple[int, int]]:
        """(perfect index, human index) of every pair reached from a selected note.

        A pair whose other end is filtered out of its roll is left out: there is nothing to draw to.
        """
        pairs = set()
        for h in self._sel["h"]:
            p = int(self.comp.human.partner[h])
            if p >= 0 and self._shown["p"][p]:
                pairs.add((p, h))
        for p in self._sel["p"]:
            h = int(self.comp.perfect.partner[p])
            if h >= 0 and self._shown["h"][h]:
                pairs.add((p, h))
        return pairs

    def _highlighted(self) -> dict[str, set[int]]:
        """Selected notes plus the far end of each drawn line."""
        pairs = self._pairs()
        return {"p": self._sel["p"] | {p for p, _ in pairs}, "h": self._sel["h"] | {h for _, h in pairs}}

    def _hit(self, x: float, y: float) -> tuple[str, int] | None:
        for item in reversed(self.canvas.find_overlapping(x - HIT_RADIUS, y - HIT_RADIUS,
                                                          x + HIT_RADIUS, y + HIT_RADIUS)):
            if item in self._items:
                return self._items[item]
        return None

    def _notes_in_box(self, x0: float, y0: float, x1: float, y1: float) -> dict[str, set[int]]:
        t_a, t_b = sorted((self._t(x0), self._t(x1)))
        y_a, y_b = sorted((y0, y1))
        found: dict[str, set[int]] = {"p": set(), "h": set()}
        for side in "ph":
            roll = self._roll(side)
            if y_b < self._top[side] or y_a > self._bottom(side):
                continue  # the box misses this roll altogether
            last = self._vis_rows - 1  # rows counted from the top of the roll, only those on screen
            row_a = min(max(int(np.floor((y_a - self._top[side]) / self._row_h)), 0), last)
            row_b = min(max(int(np.ceil((y_b - self._top[side]) / self._row_h)) - 1, 0), last)
            hit = ((roll.start <= t_b) & (roll.end >= t_a) & self._shown[side]
                   & (roll.pitches <= self._p_top - row_a) & (roll.pitches >= self._p_top - row_b))
            found[side] = {int(i) for i in np.flatnonzero(hit)}
        return found

    def _set_selection(self, selection: dict[str, set[int]]) -> None:
        self._sel = selection
        self._redraw()
        self.info.configure(text=self._summary())
        self._update_actions()

    def select_all(self) -> None:
        """Select every note the filters currently show, in both rolls."""
        self._set_selection({side: {int(i) for i in np.flatnonzero(self._shown[side])} for side in "ph"})

    def _summary(self) -> str:
        p_sel, h_sel = self._sel["p"], self._sel["h"]
        if not p_sel and not h_sel:
            return HINT
        if len(p_sel) + len(h_sel) == 1:
            side = "h" if h_sel else "p"
            return describe_note(self.comp, side, next(iter(h_sel or p_sel)))
        wrong = sum(int(self.comp.human.status[h]) == REMOVE for h in h_sel)  # still to be removed
        lonely_p = sum(int(self.comp.perfect.status[p]) not in CORRECT for p in p_sel)
        text = (f"{len(p_sel) + len(h_sel):,} notes selected ({len(p_sel):,} perfect, {len(h_sel):,} human); "
                f"{len(self._pairs()):,} matching pair(s) drawn.")
        if wrong:
            text += f" {wrong:,} selected human note(s) are wrong (red)."
        if lonely_p:
            text += f" {lonely_p:,} selected perfect note(s) are missing from the performance."
        return text

    # ---- editing --------------------------------------------------------------

    def _title_text(self) -> str:
        name = f" [{self.project_path.name}]" if self.project_path else ""
        return f"Editor: {self._title}{name}{' *' if self.session.dirty else ''}"

    def _update_title(self) -> None:
        self.title(self._title_text())

    def _update_actions(self) -> None:
        """Show or hide Clean Selection and enable the buttons that can do something now."""
        session = self.session
        remove, add = session.effect_of(self._sel["p"], self._sel["h"])
        if self._sel["p"] or self._sel["h"]:
            todo = " + ".join(part for part in (f"remove {len(remove):,}" if remove else "",
                                               f"add {len(add):,}" if add else "") if part)
            self.clean_button.configure(text=f"Clean Selection ({todo or 'nothing to do'})")
            self.clean_button.state(["!disabled" if remove or add else "disabled"])
            if not self.clean_button.winfo_manager():  # not shown yet
                self.clean_button.pack(side="right", padx=(12, 0), before=self._clean_before)
        else:
            self.clean_button.pack_forget()
        self.undo_button.state(["!disabled" if session.undo_label is not None else "disabled"])
        self.redo_button.state(["!disabled" if session.redo_label is not None else "disabled"])
        self.export_button.state(["!disabled" if session.state.count else "disabled"])
        self._update_title()

    def _edited(self, message: str) -> None:
        """The applied edits changed: redraw from the new statuses and say what happened."""
        self.comp = self.session.view()
        self._recompute_shown()
        self._sel = {side: {i for i in ids if self._shown[side][i]} for side, ids in self._sel.items()}
        self._redraw()
        self._update_actions()
        self.info.configure(text=message)

    def clean_selection(self) -> None:
        label = self.session.clean(self._sel["p"], self._sel["h"])
        if label is not None:
            self._edited(f"{label}. Ctrl+Z undoes it.")

    def undo(self) -> None:
        label = self.session.undo()
        if label is not None:
            self._edited(f"Undid: {label}.")

    def redo(self) -> None:
        label = self.session.redo()
        if label is not None:
            self._edited(f"Redid: {label}.")

    def save(self) -> bool:
        """Save the project (asking where the first time); True if it was saved."""
        path = self.project_path or dialogs.ask_project_save_path(self, self.session.human.path)
        if path is None:
            return False
        try:
            save_project(path, self.session)
        except OSError as exc:
            messagebox.showerror("Save failed", f"Could not save the project:\n{exc}", parent=self)
            return False
        self.project_path = path
        self.session.mark_saved()
        self._update_title()
        self.info.configure(text=f"Saved project {path} ({self.session.state.count:,} edit(s)).")
        return True

    def export_midi(self) -> None:
        session = self.session
        if not session.state.count:
            self.info.configure(text="Nothing to export yet: use Clean Selection first.")
            return
        try:
            cleaned = session.build_midi()
        except Exception as exc:
            messagebox.showerror("Export failed", f"Could not build the cleaned MIDI:\n{exc}", parent=self)
            return
        target = dialogs.ask_midi_save_path(self, session.human.path, "Export cleaned MIDI")
        if target is None:
            return
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            cleaned.save(str(target))
        except OSError as exc:
            messagebox.showerror("Export failed", f"Could not save the file:\n{exc}", parent=self)
            return
        state = session.state
        self.info.configure(text=f"Exported {target} (removed {len(state.removed):,} note(s), "
                                 f"added {len(state.added):,}).")

    def close(self) -> bool:
        """Close the window, offering to save unsaved edits first; False if the user backed out."""
        if self.session.dirty:
            answer = messagebox.askyesnocancel(
                "Unsaved edits", "This editor has edits that are not saved in a project.\n\n"
                "Save the project before closing?", parent=self)
            if answer is None or (answer and not self.save()):
                return False
        self.destroy()
        return True

    # ---- mouse ----------------------------------------------------------------

    def _on_press(self, event) -> None:
        self.canvas.focus_set()  # so the keyboard shortcuts reach this window
        self._press = {"x": event.x, "y": event.y, "moved": False, "extend": bool(event.state & SHIFT)}

    def _on_drag(self, event) -> None:
        press = self._press
        if press is None:
            return
        if not press["moved"] and max(abs(event.x - press["x"]), abs(event.y - press["y"])) < DRAG_THRESHOLD:
            return
        press["moved"] = True
        self.canvas.delete("band")
        self.canvas.create_rectangle(press["x"], press["y"], event.x, event.y, outline="#222222",
                                     dash=(4, 3), width=1.5, tags="band")

    def _on_release(self, event) -> None:
        press, self._press = self._press, None
        if press is None:
            return
        self.canvas.delete("band")
        base = {side: set(ids) for side, ids in self._sel.items()} if press["extend"] else {"p": set(), "h": set()}
        if press["moved"]:
            box = self._notes_in_box(press["x"], press["y"], event.x, event.y)
            self._set_selection({side: base[side] | box[side] for side in "ph"})
            return
        hit = self._hit(event.x, event.y)
        if hit is not None:
            side, index = hit
            if press["extend"] and index in base[side]:
                base[side].discard(index)  # shift-click toggles
            else:
                base[side].add(index)
        self._set_selection(base)
