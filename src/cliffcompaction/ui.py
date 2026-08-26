"""Terminal rendering helpers: colors, rules, and the wordmark banner.

Everything here degrades: no color when piped or when NO_COLOR is set, no
block glyphs when the terminal can't encode them or is too narrow.
"""

from __future__ import annotations

import os
import shutil
import sys

BRAND = (0, 210, 190)
TEXT = (203, 213, 225)
DIM = (100, 116, 139)
FAINT = (71, 85, 105)
YELLOW = (253, 224, 71)
RED = (248, 113, 113)

MAX_WIDTH = 104


def _color_enabled(stream) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM", "") == "dumb":
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def _truecolor() -> bool:
    return os.environ.get("COLORTERM", "") in ("truecolor", "24bit")


def _unicode_ok(stream) -> bool:
    enc = (getattr(stream, "encoding", None) or "").lower()
    return "utf" in enc


class Term:
    """Rendering context for one output stream."""

    def __init__(self, stream=None) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.color = _color_enabled(self.stream)
        self.unicode = _unicode_ok(self.stream)
        self.raw_columns = shutil.get_terminal_size((80, 24)).columns
        # Body text is capped so it stays readable in a very wide window; the
        # banner still uses the full width to decide its layout.
        self.columns = min(self.raw_columns, MAX_WIDTH)

    def c(self, rgb: tuple[int, int, int], s: str, bold: bool = False) -> str:
        if not self.color:
            return s
        r, g, b = rgb
        if _truecolor():
            pre = f"\033[38;2;{r};{g};{b}m"
        else:
            idx = 16 + 36 * round(r / 51) + 6 * round(g / 51) + round(b / 51)
            pre = f"\033[38;5;{idx}m"
        return pre + ("\033[1m" if bold else "") + s + "\033[0m"

    def rule(self, label: str = "", width: int | None = None) -> str:
        w = (width if width is not None else self.columns) - 2
        line = "─" if self.unicode else "-"
        if not label:
            return " " + self.c(FAINT, line * w)
        head = line * 2 + " " + label + " "
        return " " + self.c(FAINT, head + line * max(0, w - len(head)))

    def out(self, *lines: str) -> None:
        for line in lines:
            print(line, file=self.stream)


# Wordmark glyphs, 6 rows tall; "#" is substituted for the block character so
# the source stays readable in a fixed-width editor.
_GLYPHS = {
    "C": [" ######╗ ", "##╔════╝ ", "##║      ", "##║      ", "╚######╗ ", " ╚═════╝ "],
    "L": ["##╗      ", "##║      ", "##║      ", "##║      ", "#######╗ ", "╚══════╝ "],
    "I": ["##╗ ", "##║ ", "##║ ", "##║ ", "##║ ", "╚═╝ "],
    "F": ["#######╗ ", "##╔════╝ ", "#####╗   ", "##╔══╝   ", "##║      ", "╚═╝      "],
    "O": [" ######╗  ", "##╔═══##╗ ", "##║   ##║ ", "##║   ##║ ", "╚######╔╝ ", " ╚═════╝  "],
    "M": ["###╗   ###╗ ", "####╗ ####║ ", "##╔####╔##║ ", "##║╚##╔╝##║ ", "##║ ╚═╝ ##║ ", "╚═╝     ╚═╝ "],
    "P": ["######╗  ", "##╔══##╗ ", "######╔╝ ", "##╔═══╝  ", "##║      ", "╚═╝      "],
    "A": [" #####╗  ", "##╔══##╗ ", "#######║ ", "##╔══##║ ", "##║  ##║ ", "╚═╝  ╚═╝ "],
    "T": ["########╗ ", "╚══##╔══╝ ", "   ##║    ", "   ##║    ", "   ##║    ", "   ╚═╝    "],
    "N": ["###╗   ##╗ ", "####╗  ##║ ", "##╔##╗ ##║ ", "##║╚##╗##║ ", "##║ ╚####║ ", "╚═╝  ╚═══╝ "],
}


def _word(text: str) -> list[str]:
    rows = ["".join(_GLYPHS[ch][r] for ch in text) for r in range(6)]
    return [row.replace("#", "█").rstrip() for row in rows]


def banner(term: Term, indent: str = "  ") -> list[str]:
    """The CLIFFCOMPACTION wordmark, laid out for the terminal's width.

    One line when it fits, stacked when it doesn't, and a plain wordmark when
    even the stacked form would wrap (or the terminal can't render blocks).
    """
    if term.unicode:
        for rows in (_word("CLIFFCOMPACTION"), _word("CLIFF") + [""] + _word("COMPACTION")):
            if max(len(r) for r in rows) + len(indent) <= term.raw_columns:
                return [""] + [indent + term.c(BRAND, r, bold=True) if r else "" for r in rows] + [""]
    return ["", indent + term.c(BRAND, "CLIFFCOMPACTION", bold=True), ""]
