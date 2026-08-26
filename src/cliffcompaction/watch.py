"""`cliff watch`: a live view of the sessions moving through the proxy.

Reads the daemon's SSE event stream and renders one row per session. Counts
cover what this watcher has seen — the backlog it was handed on connect plus
everything since — which is the same window the sparklines are drawn from.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import time
import re
from collections import deque

from .ui import BRAND, DIM, FAINT, IDLE, TEXT, YELLOW, Term

IDLE_AFTER = 120       # seconds without traffic: the row dims
DROP_AFTER = 1800      # ...and then leaves the list
SERIES = 240           # per-session samples kept for the sparkline
FEED = 200             # events kept for the feed

BARS = "▁▂▃▄▅▆▇█"

_DATED = re.compile(r"-\d{8}$")


def short_model(name: str) -> str:
    """`claude-sonnet-4-5-20250929` -> `sonnet-4-5`: drop the vendor prefix and
    the release date, which are the same for every row that matters."""
    name = _DATED.sub("", name or "?")
    for prefix in ("claude-", "anthropic/", "openai/"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name or "?"


def kfmt(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1000:
        return f"{n / 1000:.0f}k"
    return str(int(n))


def agefmt(sec: float) -> str:
    if sec < 60:
        return f"{int(sec)}s"
    if sec < 3600:
        return f"{int(sec // 60)}m"
    return f"{int(sec // 3600)}h"


class Session:
    def __init__(self, sid: str, ev: dict) -> None:
        self.sid = sid
        self.series: deque[int] = deque(maxlen=SERIES)
        self.marks: deque[int] = deque(maxlen=SERIES)  # indices into series
        self.n = 0                                     # total samples ever
        self.compactions = 0
        self.update(ev)

    def update(self, ev: dict) -> None:
        self.model = short_model(ev.get("model") or "?")
        self.dialect = ev.get("dialect") or "?"
        self.cut = ev.get("cut") or 0
        self.total = ev.get("total") or 0
        self.est = ev.get("est_out") or ev.get("est_in") or 0
        self.last = ev.get("t") or time.time()
        self.series.append(self.est)
        if ev.get("kind") == "compact":
            self.compactions += 1
            self.marks.append(self.n)
        self.n += 1

    def spark(self, term: Term, width: int, threshold: int, idle: bool) -> str:
        pts = list(self.series)[-width:]
        first = self.n - len(pts)
        marks = {m for m in self.marks if m >= first}
        out = []
        for i, v in enumerate(pts):
            f = v / threshold if threshold else 0.0
            ch = BARS[min(max(int(f * (len(BARS) - 1) + 0.5), 0), len(BARS) - 1)]
            hot = (first + i) in marks
            out.append(term.c(IDLE if idle else (YELLOW if hot else BRAND), ch))
        return "".join(out) + " " * (width - len(pts))


class Watcher:
    def __init__(self, term: Term, port: int) -> None:
        self.term = term
        self.port = port
        self.sessions: dict[str, Session] = {}
        self.feed: deque[dict] = deque(maxlen=FEED)
        self.requests = 0
        self.compactions = 0
        self.window_start = time.time()
        self.threshold = 128_000
        self.keep_recent = 3
        self.shadow = False
        self.uptime = 0.0
        self.connected = False
        self.paused = False
        self.stop = False

    # ------------------------------------------------------------- ingest
    def on_event(self, ev: dict) -> None:
        sid = ev.get("sid")
        if not sid:
            return
        self.window_start = min(self.window_start, ev.get("t") or time.time())
        self.requests += 1
        if ev.get("kind") == "compact":
            self.compactions += 1
        s = self.sessions.get(sid)
        if s is None:
            self.sessions[sid] = Session(sid, ev)
        else:
            s.update(ev)
        self.feed.append(ev)

    def live_sessions(self) -> list[Session]:
        now = time.time()
        return sorted(
            (s for s in self.sessions.values() if now - s.last < DROP_AFTER),
            key=lambda s: s.last,
            reverse=True,
        )

    # ------------------------------------------------------------- render
    def render(self) -> str:
        t = self.term
        w = t.columns
        now = time.time()
        sessions = self.live_sessions()
        idle_n = sum(1 for s in sessions if now - s.last >= IDLE_AFTER)
        rows = max(t.rows, 20)

        head = (
            " " + t.c(BRAND, "⟨cliff⟩", bold=True) + t.c(DIM, " watch")
            + "   " + t.c(DIM, "threshold ") + t.c(TEXT, kfmt(self.threshold))
            + t.c(FAINT, " · ") + t.c(DIM, "keep ") + t.c(TEXT, str(self.keep_recent))
        )
        if self.uptime:
            head += t.c(FAINT, " · ") + t.c(DIM, "up ") + t.c(TEXT, agefmt(self.uptime))
        if self.shadow:
            head += t.c(FAINT, " · ") + t.c(YELLOW, "shadow")
        if not self.connected:
            head += "   " + t.c(YELLOW, "reconnecting…")
        elif self.paused:
            head += "   " + t.c(YELLOW, "paused")
        lines = ["", head, ""]

        sw = max(10, w - 64)
        if not sessions:
            lines += ["  " + t.c(FAINT, "no sessions yet — start an agent and it will appear here"), ""]
        for s in sessions:
            age = now - s.last
            idle = age >= IDLE_AFTER
            body = IDLE if idle else TEXT
            lines.append(
                "  " + t.c(IDLE if idle else BRAND, "○" if idle else "●")
                + " " + t.c(IDLE if idle else DIM, s.sid[:4])
                + "  " + t.c(body, s.model[:14].ljust(15))
                + t.c(FAINT, s.dialect[:9].ljust(10))
                + s.spark(t, sw, self.threshold, idle)
                + t.c(IDLE if idle else DIM,
                      "%12s" % (f"{s.cut}/{s.total}" if s.cut else f"—/{s.total}"))
                + t.c(body, "%7s" % kfmt(s.est))
                + t.c((IDLE if idle else BRAND) if s.compactions else FAINT,
                      "%5s" % (f"{s.compactions}×" if s.compactions else "—"))
                + t.c(FAINT, "%6s" % agefmt(age))
            )

        # Feed gets whatever vertical space is left.
        used = len(lines) + 8
        room = max(3, rows - used)
        lines += ["", t.rule(), ""]
        for ev in list(self.feed)[-room:]:
            lines.append(self.feed_line(ev))
        lines += ["", t.rule(), ""]
        lines.append(
            "  " + t.c(TEXT, f"{self.requests:,}") + t.c(DIM, " requests")
            + t.c(FAINT, "   ") + t.c(TEXT, str(self.compactions)) + t.c(DIM, " compactions")
            + t.c(FAINT, "   ") + t.c(TEXT, str(len(sessions))) + t.c(DIM, " sessions")
            + (t.c(DIM, f" · {idle_n} idle") if idle_n else "")
            + t.c(FAINT, "      ") + t.c(DIM, "watching ")
            + t.c(FAINT, agefmt(now - self.window_start))
        )
        lines += ["", "  " + t.c(FAINT, "p pause  ·  q quit")]
        return "\n".join(lines)

    def feed_line(self, ev: dict) -> str:
        t = self.term
        clock = time.strftime("%H:%M:%S", time.localtime(ev.get("t") or time.time()))
        sid = (ev.get("sid") or "")[:4]
        kind = ev.get("kind")
        if kind == "compact":
            steps = ev.get("steps") or 1
            body = (
                f"{kfmt(ev.get('est_in') or 0)} → {kfmt(ev.get('est_out') or 0)} est"
            ).ljust(20) + (
                f"{ev.get('total')}→{ev.get('out_msgs')} msgs   "
                f"{'replayed ' + str(steps) + '×   ' if steps > 1 else ''}"
                f"#{ev.get('fp') or '-'}"
            )
            mark = t.c(YELLOW, "⚡ compact ", bold=True)
            return "  " + t.c(DIM, clock) + " " + t.c(BRAND, sid) + "  " + mark + t.c(TEXT, body)
        if kind == "match":
            body = f"depth {ev.get('cut')}/{ev.get('total')}".ljust(20) + f"{kfmt(ev.get('est_out') or 0)} est"
            return ("  " + t.c(DIM, clock) + " " + t.c(FAINT, sid) + "  "
                    + t.c(BRAND, "· match   ") + t.c(DIM, body))
        body = f"{ev.get('total')} msgs".ljust(20) + f"{kfmt(ev.get('est_in') or 0)} est"
        return ("  " + t.c(DIM, clock) + " " + t.c(FAINT, sid) + "  "
                + t.c(FAINT, "· pass    ") + t.c(DIM, body))


async def _stream(w: Watcher) -> None:
    """Follow the daemon's event stream, reconnecting if it goes away."""
    import httpx

    url = f"http://127.0.0.1:{w.port}/__cliff__/events"
    status_url = f"http://127.0.0.1:{w.port}/__cliff__/status"
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=None)) as client:
        while not w.stop:
            try:
                with contextlib.suppress(Exception):
                    st = (await client.get(status_url)).json()
                    w.threshold = st.get("threshold_tokens", w.threshold)
                    w.keep_recent = st.get("keep_recent", w.keep_recent)
                    w.shadow = bool(st.get("shadow"))
                    w.uptime = float(st.get("uptime_s") or 0)
                async with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    w.connected = True
                    async for line in resp.aiter_lines():
                        if w.stop:
                            return
                        if line.startswith("data: "):
                            with contextlib.suppress(json.JSONDecodeError):
                                w.on_event(json.loads(line[6:]))
            except Exception:
                w.connected = False
                await asyncio.sleep(1.0)


async def _draw(w: Watcher) -> None:
    out = sys.stdout
    while not w.stop:
        if not w.paused:
            w.term.refresh()
            out.write("\033[H\033[J" + w.render())
            out.flush()
        await asyncio.sleep(1.0)


def _keys(w: Watcher) -> None:
    """q quits, p pauses. Only when stdin is a terminal."""
    try:
        ch = sys.stdin.read(1)
    except Exception:
        return
    if ch in ("q", "Q", "\x03"):
        w.stop = True
    elif ch in ("p", "P", " "):
        w.paused = not w.paused


async def _run(w: Watcher) -> None:
    loop = asyncio.get_running_loop()
    reader = False
    if sys.stdin.isatty():
        with contextlib.suppress(Exception):
            loop.add_reader(sys.stdin.fileno(), _keys, w)
            reader = True
    tasks = [asyncio.create_task(_stream(w)), asyncio.create_task(_draw(w))]
    try:
        while not w.stop:
            await asyncio.sleep(0.1)
    finally:
        if reader:
            with contextlib.suppress(Exception):
                loop.remove_reader(sys.stdin.fileno())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def run(port: int) -> int:
    import httpx

    term = Term()
    try:
        httpx.get(f"http://127.0.0.1:{port}/__cliff__/status", timeout=3.0).raise_for_status()
    except Exception:
        print(
            f"error: no cliff proxy answering on port {port}. "
            f"Run `cliff status` to check the daemon.",
            file=sys.stderr,
        )
        return 1

    w = Watcher(term, port)
    restore = None
    if sys.stdin.isatty():
        with contextlib.suppress(Exception):
            import termios
            import tty

            fd = sys.stdin.fileno()
            restore = (fd, termios.tcgetattr(fd))
            tty.setcbreak(fd)
    if term.tty:
        sys.stdout.write("\033[?1049h\033[?25l")  # alternate screen, hide cursor
    try:
        asyncio.run(_run(w))
    except KeyboardInterrupt:
        pass
    finally:
        if term.tty:
            sys.stdout.write("\033[?25h\033[?1049l")
            sys.stdout.flush()
        if restore is not None:
            with contextlib.suppress(Exception):
                import termios

                termios.tcsetattr(restore[0], termios.TCSADRAIN, restore[1])
    return 0
