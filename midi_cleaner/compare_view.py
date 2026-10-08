"""The comparison window: the perfect source's piano roll above the human performance's.

Human notes are green when they match the score, red when the score doesn't have them and grey
when the cleaner could not tell. Click a note, or drag a box around several, to draw a line to
each note's counterpart in the other roll.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

from .cleaning import DISPLACED, MATCHED, REMOVE
from .comparison import Comparison, Roll, describe_note, note_name

PERFECT_COLOR = "#5b7fa8"
CORRECT_COLOR = "#2e9e4f"
WRONG_COLOR = "#d62f2f"
UNJUDGED_COLOR = "#9a9a9a"
LINK_COLOR = "#111111"
PANEL_BG, BLACK_KEY_BG = "#fbfbfc", "#eef0f4"
GRID_COLOR, OCTAVE_COLOR = "#dfe2e8", "#b4bac6"

GUTTER = 52  # left strip with the key names
RULER = 24
TITLE = 18
GAP = 46  # between the rolls: room for the human roll's title and the lines
MARGIN = 10
MIN_PPS, MAX_PPS = 6.0, 500.0  # pixels per second
DRAG_THRESHOLD = 4
HIT_RADIUS = 2
TICK_STEPS = (0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800)
SHIFT, CONTROL = 0x1, 0x4
HINT = ("Click a note to see its match, or drag a box to select a group. "
        "Scroll to pan, Ctrl+scroll to zoom, right-drag to pan, Esc to clear.")


def _human_color(status: int) -> str:
    if status in (MATCHED, DISPLACED):
        return CORRECT_COLOR
    return WRONG_COLOR if status == REMOVE else UNJUDGED_COLOR


class ComparisonWindow(tk.Toplevel):
    def __init__(self, master: tk.Misc, comparison: Comparison, title: str) -> None:
        super().__init__(master)
        self.title(title)
        self.geometry("1180x780")
        self.minsize(640, 420)
        self.comp = comparison
        self._pps = 40.0  # pixels per second
        self._t0 = comparison.t_min  # time at the left edge of the rolls
        self._row_h = 6.0
        self._top = {"p": 0.0, "h": 0.0}  # canvas y of each roll's top edge
        self._sel: dict[str, set[int]] = {"p": set(), "h": set()}
        self._items: dict[int, tuple[str, int]] = {}  # canvas item -> (roll, note index)
        self._press: dict | None = None
        self._pan: tuple[float, float] | None = None
        self._ready = False
        self._human_colors = np.array([_human_color(int(s)) for s in comparison.human.status])

        self._build()

    # ---- widgets --------------------------------------------------------------

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        bar = ttk.Frame(self, padding=(10, 8, 10, 4))
        bar.grid(row=0, column=0, sticky="ew")
        for text, color in (("Perfect source", PERFECT_COLOR), ("Human: correct", CORRECT_COLOR),
                            ("Human: wrong", WRONG_COLOR), ("Human: not judged", UNJUDGED_COLOR)):
            tk.Label(bar, text=f" {text} ", bg=color, fg="white").pack(side="left", padx=(0, 6))
        for text, command in (("Fit", self._fit), ("+", lambda: self._zoom(1.5)),
                              ("−", lambda: self._zoom(1 / 1.5))):
            ttk.Button(bar, text=text, width=4, command=command).pack(side="right", padx=(4, 0))

        self.canvas = tk.Canvas(self, background="white", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=10)
        self.scroll = ttk.Scrollbar(self, orient="horizontal", command=self._scroll_command)
        self.scroll.grid(row=2, column=0, sticky="ew", padx=10)
        self.info = ttk.Label(self, text=HINT, wraplength=1100, justify="left", padding=(10, 6))
        self.info.grid(row=3, column=0, sticky="ew")

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
        self.info.bind("<Configure>", lambda e: self.info.configure(wraplength=max(e.width - 20, 200)))

    # ---- geometry -------------------------------------------------------------

    def _x(self, t):
        return GUTTER + (t - self._t0) * self._pps

    def _t(self, x):
        return self._t0 + (x - GUTTER) / self._pps

    def _y(self, side: str, pitch):
        return self._top[side] + (self.comp.pitch_hi - pitch) * self._row_h

    def _span(self) -> float:
        """Seconds of music visible across the rolls."""
        return max(self.canvas.winfo_width() - GUTTER, 1) / self._pps

    def _bounds(self) -> tuple[float, float]:
        return self.comp.t_min - 1.0, self.comp.t_max + 1.0

    def _clamp_view(self) -> None:
        lo, hi = self._bounds()
        self._t0 = min(max(self._t0, lo), max(hi - self._span(), lo))

    def _layout(self) -> None:
        rows = self.comp.pitch_hi - self.comp.pitch_lo + 1
        free = self.canvas.winfo_height() - RULER - TITLE - GAP - MARGIN
        self._row_h = float(np.clip(free / (2 * rows), 1.5, 10.0))
        self._top["p"] = RULER + TITLE
        self._top["h"] = self._top["p"] + rows * self._row_h + GAP

    def _on_resize(self, _event) -> None:
        self._layout()
        if not self._ready:
            self._ready = True
            self._fit()
            return
        self._clamp_view()
        self._redraw()

    # ---- view changes ---------------------------------------------------------

    def _fit(self) -> None:
        """Show the whole piece if it is short enough to read, else its first minute or so."""
        lo, hi = self._bounds()
        width = max(self.canvas.winfo_width() - GUTTER, 1)
        self._pps = float(np.clip(width / (hi - lo), MIN_PPS, MAX_PPS))
        self._t0 = lo
        self._clamp_view()
        self._redraw()

    def _zoom(self, factor: float, anchor_x: float | None = None) -> None:
        if anchor_x is None:
            anchor_x = GUTTER + (self.canvas.winfo_width() - GUTTER) / 2
        anchor_t = self._t(anchor_x)
        self._pps = float(np.clip(self._pps * factor, MIN_PPS, MAX_PPS))
        self._t0 = anchor_t - (anchor_x - GUTTER) / self._pps
        self._clamp_view()
        self._redraw()

    def _scroll_command(self, kind: str, *args: str) -> None:
        lo, hi = self._bounds()
        if kind == "moveto":
            self._t0 = lo + float(args[0]) * (hi - lo)
        elif kind == "scroll":
            step = self._span() * (0.9 if args[1] == "pages" else 0.1)
            self._t0 += int(args[0]) * step
        self._clamp_view()
        self._redraw()

    def _on_wheel(self, event) -> None:
        if event.num == 4 or (event.num != 5 and event.delta > 0):
            direction = 1
        else:
            direction = -1
        if event.state & CONTROL:
            self._zoom(1.25 ** direction, event.x)
        else:
            self._t0 -= direction * self._span() * 0.1
            self._clamp_view()
            self._redraw()

    def _on_pan_start(self, event) -> None:
        self._pan = (event.x, self._t0)

    def _on_pan_move(self, event) -> None:
        if self._pan is not None:
            self._t0 = self._pan[1] - (event.x - self._pan[0]) / self._pps
            self._clamp_view()
            self._redraw()

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
        self._draw_links()
        self._update_scrollbar()

    def _draw_panel(self, side: str, title: str, width: int) -> None:
        c, comp = self.canvas, self.comp
        top, bottom = self._top[side], self._top[side] + (comp.pitch_hi - comp.pitch_lo + 1) * self._row_h
        c.create_text(GUTTER, top - 4, text=title, anchor="sw", fill="#444444", font=("TkDefaultFont", 9, "bold"))
        c.create_rectangle(GUTTER, top, width, bottom, fill=PANEL_BG, outline=GRID_COLOR)
        for pitch in range(comp.pitch_lo, comp.pitch_hi + 1):
            y = self._y(side, pitch)
            if pitch % 12 in (1, 3, 6, 8, 10):
                c.create_rectangle(GUTTER, y, width, y + self._row_h, fill=BLACK_KEY_BG, outline="")
            if pitch % 12 == 0:
                c.create_line(GUTTER, y + self._row_h, width, y + self._row_h, fill=OCTAVE_COLOR)
                c.create_text(GUTTER - 6, y + self._row_h / 2, text=note_name(pitch), anchor="e",
                              fill="#555555", font=("TkDefaultFont", 8))

    def _draw_time_grid(self, width: int, t_a: float, t_b: float) -> None:
        c = self.canvas
        step = next((s for s in TICK_STEPS if s * self._pps >= 70), TICK_STEPS[-1])
        first = int(np.ceil(max(t_a, 0.0) / step))
        bottoms = {side: self._top[side] + (self.comp.pitch_hi - self.comp.pitch_lo + 1) * self._row_h
                   for side in "ph"}
        for k in range(first, int(t_b / step) + 1):
            t = k * step
            x = self._x(t)
            for side in "ph":
                c.create_line(x, self._top[side], x, bottoms[side], fill=GRID_COLOR)
            label = f"{int(t // 60)}:{t % 60:02.0f}" if step >= 1 else f"{t:.2f}s"
            c.create_line(x, RULER - 5, x, RULER, fill=OCTAVE_COLOR)
            c.create_text(x + 3, RULER - 6, text=label, anchor="sw", fill="#555555", font=("TkDefaultFont", 8))

    def _draw_notes(self, side: str, t_a: float, t_b: float, highlight: dict[str, set[int]]) -> None:
        roll = self._roll(side)
        visible = np.flatnonzero((roll.end >= t_a) & (roll.start <= t_b))
        x1 = self._x(roll.start[visible])
        x2 = np.maximum(self._x(roll.end[visible]), x1 + 2.0)
        y1 = self._y(side, roll.pitches[visible]) + (0.5 if self._row_h >= 4 else 0.0)
        height = self._row_h - (1.0 if self._row_h >= 4 else 0.0)
        colors = self._human_colors[visible] if side == "h" else None
        marked = highlight[side]
        create, items = self.canvas.create_rectangle, self._items
        for n, i in enumerate(visible):
            i = int(i)
            on = i in marked
            item = create(x1[n], y1[n], x2[n], y1[n] + height,
                          fill=colors[n] if colors is not None else PERFECT_COLOR,
                          outline="black" if on else "", width=1)
            items[item] = (side, i)

    def _draw_links(self) -> None:
        comp = self.comp
        for p, h in sorted(self._pairs()):
            xp = self._x((comp.perfect.start[p] + comp.perfect.end[p]) / 2)
            xh = self._x((comp.human.start[h] + comp.human.end[h]) / 2)
            yp = self._y("p", comp.perfect.pitches[p]) + self._row_h / 2
            yh = self._y("h", comp.human.pitches[h]) + self._row_h / 2
            self.canvas.create_line(xp, yp, xh, yh, fill=LINK_COLOR, width=1.5)

    def _update_scrollbar(self) -> None:
        lo, hi = self._bounds()
        first = (self._t0 - lo) / (hi - lo)
        self.scroll.set(first, min(first + self._span() / (hi - lo), 1.0))

    # ---- selection ------------------------------------------------------------

    def _roll(self, side: str) -> Roll:
        return self.comp.human if side == "h" else self.comp.perfect

    def _pairs(self) -> set[tuple[int, int]]:
        """(perfect index, human index) of every pair reached from a selected note."""
        pairs = set()
        for h in self._sel["h"]:
            p = int(self.comp.human.partner[h])
            if p >= 0:
                pairs.add((p, h))
        for p in self._sel["p"]:
            h = int(self.comp.perfect.partner[p])
            if h >= 0:
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
            row_a = int(np.floor((y_a - self._top[side]) / self._row_h))  # rows counted from the top
            row_b = int(np.ceil((y_b - self._top[side]) / self._row_h)) - 1
            hit = ((roll.start <= t_b) & (roll.end >= t_a)
                   & (roll.pitches <= self.comp.pitch_hi - row_a) & (roll.pitches >= self.comp.pitch_hi - row_b))
            found[side] = {int(i) for i in np.flatnonzero(hit)}
        return found

    def _set_selection(self, selection: dict[str, set[int]]) -> None:
        self._sel = selection
        self._redraw()
        self.info.configure(text=self._summary())

    def _summary(self) -> str:
        p_sel, h_sel = self._sel["p"], self._sel["h"]
        if not p_sel and not h_sel:
            return HINT
        if len(p_sel) + len(h_sel) == 1:
            side = "h" if h_sel else "p"
            return describe_note(self.comp, side, next(iter(h_sel or p_sel)))
        wrong = sum(int(self.comp.human.status[h]) == REMOVE for h in h_sel)
        lonely_p = sum(int(self.comp.perfect.partner[p]) < 0 for p in p_sel)
        text = (f"{len(p_sel) + len(h_sel):,} notes selected ({len(p_sel):,} perfect, {len(h_sel):,} human); "
                f"{len(self._pairs()):,} matching pair(s) drawn.")
        if wrong:
            text += f" {wrong:,} selected human note(s) are wrong (red)."
        if lonely_p:
            text += f" {lonely_p:,} selected perfect note(s) have no counterpart in the performance."
        return text

    # ---- mouse ----------------------------------------------------------------

    def _on_press(self, event) -> None:
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
