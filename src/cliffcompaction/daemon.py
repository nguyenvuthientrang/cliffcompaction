"""cliff enable / disable / status: user-service install (launchd /
systemd --user) plus ANTHROPIC_BASE_URL / OPENAI_BASE_URL wiring in the
shell profile. The service runs `<python> -m cliffcompaction.cli serve ...`
with the absolute interpreter path captured at enable time, so PATH does
not matter.

Instances: the default (unnamed) daemon owns the shell env wiring. Any
number of named instances can be installed alongside it, each with its own
port, flags, service label and log; clients that cannot follow the env
(a Codex model provider, a second upstream) are pointed at a named
instance's port directly. `name=None` everywhere means the default.
"""

from __future__ import annotations

import os
import plistlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

LABEL = "com.cliffcompaction.proxy"
SYSTEMD_UNIT = "cliffcompaction.service"
MARK_BEGIN = "# >>> cliffcompaction >>>"
MARK_END = "# <<< cliffcompaction <<<"

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")


# --- instances ------------------------------------------------------------------


def validate_name(name: str | None) -> str | None:
    """Return the name unchanged, or raise: it becomes part of a launchd label,
    a systemd unit name and a log file name, so keep it to a safe charset."""
    if name is None or name == "":
        return None
    if not _NAME_RE.match(name):
        raise ValueError(
            f"invalid instance name {name!r}: use letters, digits, '-' or '_' "
            "(max 32 chars)"
        )
    return name


def label(name: str | None = None) -> str:
    return LABEL if name is None else f"{LABEL}.{name}"


def systemd_unit(name: str | None = None) -> str:
    return SYSTEMD_UNIT if name is None else f"cliffcompaction-{name}.service"


def installed_names() -> list[str | None]:
    """Every installed instance, default first (as None), then named ones sorted."""
    names: list[str | None] = []
    if sys.platform == "darwin":
        folder = Path.home() / "Library" / "LaunchAgents"
        if plist_path().exists():
            names.append(None)
        for p in sorted(folder.glob(f"{LABEL}.*.plist")):
            names.append(p.name[len(LABEL) + 1 : -len(".plist")])
    elif sys.platform.startswith("linux"):
        folder = Path.home() / ".config" / "systemd" / "user"
        if systemd_unit_path().exists():
            names.append(None)
        for p in sorted(folder.glob("cliffcompaction-*.service")):
            names.append(p.name[len("cliffcompaction-") : -len(".service")])
    return names


def installed_port(name: str | None = None) -> int | None:
    """The --port baked into the installed service definition, if any."""
    argv: list[str] | None = None
    if sys.platform == "darwin":
        path = plist_path(name)
        if path.exists():
            try:
                argv = plistlib.loads(path.read_bytes()).get("ProgramArguments")
            except Exception:
                argv = None
    elif sys.platform.startswith("linux"):
        path = systemd_unit_path(name)
        if path.exists():
            for line in path.read_text().splitlines():
                if line.startswith("ExecStart="):
                    argv = line[len("ExecStart=") :].split()
    if not argv:
        return None
    for i, tok in enumerate(argv):
        if tok == "--port" and i + 1 < len(argv):
            try:
                return int(argv[i + 1])
            except ValueError:
                return None
    return None


# --- shell profile wiring -----------------------------------------------------


def default_profile() -> Path:
    shell = Path(os.environ.get("SHELL", "")).name
    home = Path.home()
    if shell == "fish":
        return home / ".config" / "fish" / "conf.d" / "cliffcompaction.fish"
    if shell == "bash":
        return home / ".bashrc"
    return home / ".zshrc"


def env_block(port: int, fish: bool) -> str:
    base = f"http://127.0.0.1:{port}"
    if fish:
        lines = [
            f'set -gx ANTHROPIC_BASE_URL "{base}"',
            f'set -gx OPENAI_BASE_URL "{base}/v1"',
            f'set -gx OPENAI_API_BASE "{base}/v1"',
        ]
    else:
        lines = [
            f'export ANTHROPIC_BASE_URL="{base}"',
            f'export OPENAI_BASE_URL="{base}/v1"',
            f'export OPENAI_API_BASE="{base}/v1"',
        ]
    return MARK_BEGIN + "\n" + "\n".join(lines) + "\n" + MARK_END + "\n"


def _strip_block(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    inside = False
    for line in lines:
        if line.strip() == MARK_BEGIN:
            inside = True
            continue
        if line.strip() == MARK_END:
            inside = False
            continue
        if not inside:
            out.append(line)
    return "".join(out)


def wire_profile(profile: Path, port: int) -> None:
    """Add (or refresh) the env block. Idempotent."""
    fish = profile.suffix == ".fish"
    profile.parent.mkdir(parents=True, exist_ok=True)
    if fish:
        # fish conf.d: the whole file is ours.
        profile.write_text(env_block(port, fish=True))
        return
    text = profile.read_text() if profile.exists() else ""
    text = _strip_block(text)
    if text and not text.endswith("\n"):
        text += "\n"
    profile.write_text(text + env_block(port, fish=False))


def unwire_profile(profile: Path) -> None:
    if not profile.exists():
        return
    if profile.suffix == ".fish":
        profile.unlink()
        return
    profile.write_text(_strip_block(profile.read_text()))


def profile_is_wired(profile: Path) -> bool:
    if not profile.exists():
        return False
    return MARK_BEGIN in profile.read_text()


# --- service definitions --------------------------------------------------------


def serve_argv(port: int, serve_args: list[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "cliffcompaction.cli",
        "serve",
        "--port",
        str(port),
        *serve_args,
    ]


def log_path(name: str | None = None) -> Path:
    suffix = "" if name is None else f"-{name}"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / f"cliffcompaction{suffix}.log"
    return Path.home() / ".local" / "state" / "cliffcompaction" / f"proxy{suffix}.log"


def plist_path(name: str | None = None) -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{label(name)}.plist"


def plist_content(port: int, serve_args: list[str], name: str | None = None) -> bytes:
    log = str(log_path(name))
    data = {
        "Label": label(name),
        "ProgramArguments": serve_argv(port, serve_args),
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": log,
        "StandardErrorPath": log,
    }
    return plistlib.dumps(data)


def systemd_unit_path(name: str | None = None) -> Path:
    return Path.home() / ".config" / "systemd" / "user" / systemd_unit(name)


def systemd_unit_content(port: int, serve_args: list[str], name: str | None = None) -> str:
    exec_start = " ".join(serve_argv(port, serve_args))
    desc = "CliffCompaction proxy" if name is None else f"CliffCompaction proxy ({name})"
    return (
        "[Unit]\n"
        f"Description={desc}\n\n"
        "[Service]\n"
        f"ExecStart={exec_start}\n"
        "Restart=always\n"
        "RestartSec=1\n\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


# --- service control --------------------------------------------------------------


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def _gui_domain() -> str:
    return f"gui/{os.getuid()}"


def start_service(port: int, serve_args: list[str], name: str | None = None) -> None:
    lbl = label(name)
    unit = systemd_unit(name)
    if sys.platform == "darwin":
        path = plist_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        log_path(name).parent.mkdir(parents=True, exist_ok=True)
        # Refresh: bootout any existing instance first (ignore failures).
        _run(["launchctl", "bootout", f"{_gui_domain()}/{lbl}"])
        path.write_bytes(plist_content(port, serve_args, name))
        # bootout is asynchronous: launchd may still hold the label when the
        # bootstrap arrives, which fails with "Bootstrap failed: 5: Input/
        # output error". Retry briefly instead of surfacing that to the user.
        res = _run(["launchctl", "bootstrap", _gui_domain(), str(path)])
        for _ in range(10):
            if res.returncode == 0:
                break
            time.sleep(0.3)
            _run(["launchctl", "bootout", f"{_gui_domain()}/{lbl}"])
            res = _run(["launchctl", "bootstrap", _gui_domain(), str(path)])
        if res.returncode != 0:
            raise RuntimeError(f"launchctl bootstrap failed: {res.stderr.strip()}")
    elif sys.platform.startswith("linux"):
        path = systemd_unit_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(systemd_unit_content(port, serve_args, name))
        for cmd in (
            ["systemctl", "--user", "daemon-reload"],
            ["systemctl", "--user", "enable", "--now", unit],
            ["systemctl", "--user", "restart", unit],
        ):
            res = _run(cmd)
            if res.returncode != 0:
                raise RuntimeError(f"{' '.join(cmd)} failed: {res.stderr.strip()}")
    else:
        raise RuntimeError(
            f"unsupported platform for `cliff enable`: {sys.platform} "
            "(use `cliff serve` or `cliff run` instead)"
        )


def restart_service(name: str | None = None) -> None:
    """Restart the installed service in place, leaving the plist/unit alone.

    Distinct from `enable`, which rewrites the service definition: the flags
    are baked in there, so restarting must not touch it or an upgrade would
    silently reset them.
    """
    if sys.platform == "darwin":
        res = _run(["launchctl", "kickstart", "-k", f"{_gui_domain()}/{label(name)}"])
        if res.returncode != 0:
            raise RuntimeError(f"launchctl kickstart failed: {res.stderr.strip()}")
    elif sys.platform.startswith("linux"):
        res = _run(["systemctl", "--user", "restart", systemd_unit(name)])
        if res.returncode != 0:
            raise RuntimeError(f"systemctl restart failed: {res.stderr.strip()}")
    else:
        raise RuntimeError(f"unsupported platform: {sys.platform}")


def stop_service(name: str | None = None) -> None:
    if sys.platform == "darwin":
        _run(["launchctl", "bootout", f"{_gui_domain()}/{label(name)}"])
        plist_path(name).unlink(missing_ok=True)
    elif sys.platform.startswith("linux"):
        _run(["systemctl", "--user", "disable", "--now", systemd_unit(name)])
        systemd_unit_path(name).unlink(missing_ok=True)
        _run(["systemctl", "--user", "daemon-reload"])


def service_installed(name: str | None = None) -> bool:
    if sys.platform == "darwin":
        return plist_path(name).exists()
    if sys.platform.startswith("linux"):
        return systemd_unit_path(name).exists()
    return False


def service_running(name: str | None = None) -> tuple[bool, int | None]:
    """(running, last_exit_status) for the installed service.

    A plist/unit on disk says nothing about the service actually running: a
    stale proxy squatting the port makes the daemon die on bind while
    probe() still gets healthy answers from the squatter.
    """
    if sys.platform == "darwin":
        lbl = label(name)
        res = _run(["launchctl", "list"])
        for line in res.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) >= 3 and parts[2].strip() == lbl:
                pid = None if parts[0].strip() == "-" else int(parts[0])
                try:
                    last = int(parts[1])
                except ValueError:
                    last = None
                return pid is not None, last
        return False, None
    if sys.platform.startswith("linux"):
        res = _run(["systemctl", "--user", "is-active", systemd_unit(name)])
        return res.stdout.strip() == "active", None
    return False, None


def port_in_use(port: int) -> bool:
    """True if something already listens on the loopback port."""
    import socket

    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return True
    return False


# --- health -------------------------------------------------------------------------


def probe(port: int, timeout: float = 1.0) -> dict | None:
    """GET the proxy status endpoint; None if unreachable."""
    import json

    url = f"http://127.0.0.1:{port}/__cliff__/status"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.load(resp)
    except (urllib.error.URLError, OSError, ValueError):
        return None


def wait_healthy(port: int, deadline_s: float = 8.0) -> dict | None:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        status = probe(port)
        if status is not None:
            return status
        time.sleep(0.15)
    return None
