"""Daemon management: profile wiring and service definitions (no launchctl/systemctl)."""

import plistlib
import sys

from cliffcompaction import daemon


def test_env_block_posix():
    block = daemon.env_block(8257, fish=False)
    assert 'export ANTHROPIC_BASE_URL="http://127.0.0.1:8257"' in block
    assert 'export OPENAI_BASE_URL="http://127.0.0.1:8257/v1"' in block
    assert block.startswith(daemon.MARK_BEGIN)
    assert block.rstrip().endswith(daemon.MARK_END)


def test_env_block_fish():
    block = daemon.env_block(9000, fish=True)
    assert 'set -gx ANTHROPIC_BASE_URL "http://127.0.0.1:9000"' in block


def test_wire_profile_idempotent(tmp_path):
    profile = tmp_path / ".zshrc"
    profile.write_text("# my rc\nexport FOO=1\n")
    daemon.wire_profile(profile, 8257)
    daemon.wire_profile(profile, 9001)  # refresh with a new port
    text = profile.read_text()
    assert text.count(daemon.MARK_BEGIN) == 1
    assert "9001" in text and "8257" not in text
    assert "export FOO=1" in text  # untouched user content
    assert daemon.profile_is_wired(profile)


def test_unwire_profile_clean(tmp_path):
    profile = tmp_path / ".zshrc"
    profile.write_text("export FOO=1\n")
    daemon.wire_profile(profile, 8257)
    daemon.unwire_profile(profile)
    assert profile.read_text() == "export FOO=1\n"
    assert not daemon.profile_is_wired(profile)


def test_fish_profile_is_own_file(tmp_path):
    profile = tmp_path / "conf.d" / "cliffcompaction.fish"
    daemon.wire_profile(profile, 8257)
    assert profile.exists()
    assert "set -gx" in profile.read_text()
    daemon.unwire_profile(profile)
    assert not profile.exists()


def test_plist_content():
    data = plistlib.loads(daemon.plist_content(8257, ["--shadow"]))
    assert data["Label"] == daemon.LABEL
    assert data["ProgramArguments"][0] == sys.executable
    assert "-m" in data["ProgramArguments"]
    assert "cliffcompaction.cli" in data["ProgramArguments"]
    assert "serve" in data["ProgramArguments"]
    assert "8257" in data["ProgramArguments"]
    assert "--shadow" in data["ProgramArguments"]
    assert data["KeepAlive"] is True
    assert data["RunAtLoad"] is True


def test_systemd_unit_content():
    unit = daemon.systemd_unit_content(8257, [])
    assert f"ExecStart={sys.executable} -m cliffcompaction.cli serve --port 8257" in unit
    assert "Restart=always" in unit


def test_probe_unreachable():
    assert daemon.probe(1, timeout=0.2) is None


def test_port_in_use_detects_listener():
    import socket

    from cliffcompaction import daemon

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.listen(1)
    try:
        assert daemon.port_in_use(port) is True
    finally:
        s.close()
    assert daemon.port_in_use(port) is False


def test_service_running_shape():
    from cliffcompaction import daemon

    running, last_exit = daemon.service_running()
    assert isinstance(running, bool)
    assert last_exit is None or isinstance(last_exit, int)


def test_every_common_flag_reaches_the_daemon():
    """`cliff enable --foo` bakes flags into the service definition, so a flag
    accepted by the parser but dropped by _serve_args_from is silently ignored."""
    import argparse

    from cliffcompaction import cli

    parser = argparse.ArgumentParser(add_help=False)
    cli._add_common_flags(parser)

    argv = []
    for action in parser._actions:
        if not action.option_strings:
            continue
        flag = max(action.option_strings, key=len)
        argv.append(flag)
        if action.nargs != 0:
            argv.append("7" if action.type is int else "x")

    forwarded = cli._serve_args_from(parser.parse_args(argv))
    dropped = [
        a.option_strings[-1]
        for a in parser._actions
        if a.option_strings and not any(o in forwarded for o in a.option_strings)
    ]
    assert not dropped, f"accepted but never forwarded to the daemon: {dropped}"


def test_restart_needs_an_installed_daemon(monkeypatch, capsys):
    # `restart` is for picking up an upgrade, not for standing a daemon up.
    import argparse

    from cliffcompaction import cli, daemon

    monkeypatch.setattr(daemon, "service_installed", lambda: False)
    rc = cli.cmd_restart(argparse.Namespace(port=None))
    assert rc == 1
    assert "cliff enable" in capsys.readouterr().err


def test_restart_leaves_the_service_definition_alone(monkeypatch):
    # The daemon's flags live in the plist/unit. Restarting must not rewrite
    # it, or an upgrade would quietly reset --threshold and friends.
    from cliffcompaction import daemon

    calls = []
    monkeypatch.setattr(daemon, "_run", lambda cmd: calls.append(cmd) or _ok())
    monkeypatch.setattr(daemon.sys, "platform", "darwin")
    written = []
    monkeypatch.setattr(daemon.Path, "write_bytes", lambda self, b: written.append(self))
    daemon.restart_service()
    assert written == []
    assert any("kickstart" in " ".join(c) for c in calls)


def _ok():
    import subprocess

    return subprocess.CompletedProcess([], 0, "", "")


def test_status_reports_stale_code():
    """A daemon keeps serving the code it started with; nothing about a
    healthy old process says so."""
    import httpx
    from starlette.testclient import TestClient

    from cliffcompaction.config import Config
    from cliffcompaction.proxy import code_mtime, create_app

    app = create_app(
        Config.from_env(),
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
    )
    assert TestClient(app).get("/__cliff__/status").json()["code_stale"] is False
    assert code_mtime() > 0
