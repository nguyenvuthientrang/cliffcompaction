"""cliff enable / disable / status: user-service install (launchd /
systemd --user) plus ANTHROPIC_BASE_URL / OPENAI_BASE_URL wiring in the
shell profile. The service runs `<python> -m cliffcompaction.cli serve ...`
with the absolute interpreter path captured at enable time, so PATH does
not matter.
"""

from __future__ import annotations

import os
import plistlib
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


def log_path() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "cliffcompaction.log"
    return Path.home() / ".local" / "state" / "cliffcompaction" / "proxy.log"


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def plist_content(port: int, serve_args: list[str]) -> bytes:
    log = str(log_path())
    data = {
        "Label": LABEL,
        "ProgramArguments": serve_argv(port, serve_args),
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": log,
        "StandardErrorPath": log,
    }
    return plistlib.dumps(data)


def systemd_unit_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / SYSTEMD_UNIT


def systemd_unit_content(port: int, serve_args: list[str]) -> str:
    exec_start = " ".join(serve_argv(port, serve_args))
    return (
        "[Unit]\n"
        "Description=CliffCompaction proxy\n\n"
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


def start_service(port: int, serve_args: list[str]) -> None:
    if sys.platform == "darwin":
        path = plist_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        log_path().parent.mkdir(parents=True, exist_ok=True)
        # Refresh: bootout any existing instance first (ignore failures).
        _run(["launchctl", "bootout", f"{_gui_domain()}/{LABEL}"])
        path.write_bytes(plist_content(port, serve_args))
        res = _run(["launchctl", "bootstrap", _gui_domain(), str(path)])
        if res.returncode != 0:
            raise RuntimeError(f"launchctl bootstrap failed: {res.stderr.strip()}")
    elif sys.platform.startswith("linux"):
        path = systemd_unit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(systemd_unit_content(port, serve_args))
        for cmd in (
            ["systemctl", "--user", "daemon-reload"],
            ["systemctl", "--user", "enable", "--now", SYSTEMD_UNIT],
            ["systemctl", "--user", "restart", SYSTEMD_UNIT],
        ):
            res = _run(cmd)
            if res.returncode != 0:
                raise RuntimeError(f"{' '.join(cmd)} failed: {res.stderr.strip()}")
    else:
        raise RuntimeError(
            f"unsupported platform for `cliff enable`: {sys.platform} "
            "(use `cliff serve` or `cliff run` instead)"
        )


def stop_service() -> None:
    if sys.platform == "darwin":
        _run(["launchctl", "bootout", f"{_gui_domain()}/{LABEL}"])
        plist_path().unlink(missing_ok=True)
    elif sys.platform.startswith("linux"):
        _run(["systemctl", "--user", "disable", "--now", SYSTEMD_UNIT])
        systemd_unit_path().unlink(missing_ok=True)
        _run(["systemctl", "--user", "daemon-reload"])


def service_installed() -> bool:
    if sys.platform == "darwin":
        return plist_path().exists()
    if sys.platform.startswith("linux"):
        return systemd_unit_path().exists()
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
