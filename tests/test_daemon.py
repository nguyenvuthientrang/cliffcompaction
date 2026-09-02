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


def test_default_instance_paths_unchanged():
    # name=None must produce exactly what the single-daemon versions produced,
    # or every existing install's plist/unit/log would move on upgrade.
    assert daemon.label() == daemon.LABEL == "com.cliffcompaction.proxy"
    assert daemon.systemd_unit() == daemon.SYSTEMD_UNIT == "cliffcompaction.service"
    assert daemon.plist_path().name == "com.cliffcompaction.proxy.plist"
    assert daemon.log_path().name in ("cliffcompaction.log", "proxy.log")
    assert plistlib.loads(daemon.plist_content(8257, []))["Label"] == daemon.LABEL


def test_named_instance_gets_its_own_label_unit_and_log():
    assert daemon.label("codex") == "com.cliffcompaction.proxy.codex"
    assert daemon.systemd_unit("codex") == "cliffcompaction-codex.service"
    assert daemon.plist_path("codex").name == "com.cliffcompaction.proxy.codex.plist"
    assert daemon.log_path("codex").name in ("cliffcompaction-codex.log", "proxy-codex.log")
    assert daemon.log_path("codex") != daemon.log_path()

    data = plistlib.loads(daemon.plist_content(8398, ["--threshold", "200000"], "codex"))
    assert data["Label"] == "com.cliffcompaction.proxy.codex"
    assert data["StandardOutPath"] == str(daemon.log_path("codex"))
    assert "8398" in data["ProgramArguments"]

    unit = daemon.systemd_unit_content(8398, [], "codex")
    assert "Description=CliffCompaction proxy (codex)" in unit
    assert "--port 8398" in unit


def test_validate_name():
    import pytest

    assert daemon.validate_name(None) is None
    assert daemon.validate_name("") is None
    assert daemon.validate_name("codex") == "codex"
    assert daemon.validate_name("kimi-k3_2") == "kimi-k3_2"
    for bad in ("with space", "a/b", ".hidden", "-lead", "x" * 33, "ünïcode"):
        with pytest.raises(ValueError):
            daemon.validate_name(bad)


def test_installed_port_reads_the_service_definition(monkeypatch, tmp_path):
    # status/restart/watch --name X must find X's port without the user
    # repeating --port; the only durable record of it is the plist/unit.
    monkeypatch.setattr(daemon.Path, "home", classmethod(lambda cls: tmp_path))

    monkeypatch.setattr(daemon.sys, "platform", "darwin")
    daemon.plist_path("codex").parent.mkdir(parents=True)
    daemon.plist_path("codex").write_bytes(daemon.plist_content(8398, ["-v"], "codex"))
    assert daemon.installed_port("codex") == 8398
    assert daemon.installed_port() is None  # default not installed here

    monkeypatch.setattr(daemon.sys, "platform", "linux")
    daemon.systemd_unit_path("kimi").parent.mkdir(parents=True)
    daemon.systemd_unit_path("kimi").write_text(daemon.systemd_unit_content(8400, [], "kimi"))
    assert daemon.installed_port("kimi") == 8400


def test_installed_names_lists_default_first_then_named(monkeypatch, tmp_path):
    monkeypatch.setattr(daemon.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(daemon.sys, "platform", "darwin")
    folder = daemon.plist_path().parent
    folder.mkdir(parents=True)
    assert daemon.installed_names() == []
    daemon.plist_path("kimi").write_bytes(daemon.plist_content(8400, [], "kimi"))
    daemon.plist_path("codex").write_bytes(daemon.plist_content(8398, [], "codex"))
    assert daemon.installed_names() == ["codex", "kimi"]
    daemon.plist_path().write_bytes(daemon.plist_content(8257, []))
    assert daemon.installed_names() == [None, "codex", "kimi"]
    # unrelated plists in LaunchAgents are not instances
    (folder / "com.example.other.plist").write_bytes(b"")
    assert daemon.installed_names() == [None, "codex", "kimi"]


def test_service_control_targets_the_named_label(monkeypatch):
    from cliffcompaction import daemon

    calls = []
    monkeypatch.setattr(daemon, "_run", lambda cmd: calls.append(cmd) or _ok())
    monkeypatch.setattr(daemon.sys, "platform", "darwin")
    daemon.restart_service("codex")
    assert calls[-1][-1].endswith("/com.cliffcompaction.proxy.codex")

    monkeypatch.setattr(daemon.Path, "unlink", lambda self, missing_ok=False: None)
    daemon.stop_service("codex")
    assert calls[-1][:2] == ["launchctl", "bootout"]
    assert calls[-1][-1].endswith("/com.cliffcompaction.proxy.codex")

    monkeypatch.setattr(daemon.sys, "platform", "linux")
    daemon.restart_service("codex")
    assert calls[-1] == ["systemctl", "--user", "restart", "cliffcompaction-codex.service"]


def _enable_args(**over):
    import argparse

    base = dict(
        port=None, name=None, profile=None, no_env=False, shadow=False, strict=False,
        threshold=None, keep_recent=None, thought_max_chars=None, result_max_chars=None,
        drop_thinking=False, thinking_max_chars=None, anthropic_upstream=None,
        openai_upstream=None, debug_dir=None, verbose=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_enable_named_requires_a_port(capsys):
    from cliffcompaction import cli

    assert cli.cmd_enable(_enable_args(name="codex")) == 1
    assert "--port" in capsys.readouterr().err


def test_enable_rejects_bad_name(capsys):
    from cliffcompaction import cli

    assert cli.cmd_enable(_enable_args(name="bad name", port=8398)) == 1
    assert "invalid instance name" in capsys.readouterr().err


def test_enable_named_installs_alongside_and_never_wires_env(monkeypatch, tmp_path):
    # The default daemon owns ANTHROPIC_BASE_URL/OPENAI_BASE_URL; a named one
    # must not steal them, even without --no-env.
    from cliffcompaction import cli, daemon

    started, wired = [], []
    monkeypatch.setattr(daemon, "service_running", lambda name=None: (False, None))
    monkeypatch.setattr(daemon, "port_in_use", lambda port: False)
    monkeypatch.setattr(daemon, "start_service", lambda port, a, name=None: started.append((port, a, name)))
    monkeypatch.setattr(daemon, "wait_healthy", lambda port: {"threshold_tokens": 200000, "keep_recent": 3})
    monkeypatch.setattr(daemon, "wire_profile", lambda profile, port: wired.append(port))
    monkeypatch.setattr(cli, "_print_enable_screen", lambda *a, **k: None)

    rc = cli.cmd_enable(_enable_args(name="codex", port=8398, openai_upstream="https://chatgpt.com"))
    assert rc == 0
    assert started == [(8398, ["--openai-upstream", "https://chatgpt.com"], "codex")]
    assert wired == []

    rc = cli.cmd_enable(_enable_args(threshold=210000))
    assert rc == 0
    assert started[-1] == (8257, ["--threshold", "210000"], None)
    assert wired == [8257]


def test_resolve_port_prefers_flag_then_installed_then_default(monkeypatch):
    import argparse

    from cliffcompaction import cli, daemon

    monkeypatch.setattr(daemon, "installed_port", lambda name=None: 8398 if name == "codex" else None)
    assert cli._resolve_port(argparse.Namespace(port=9999), "codex") == 9999
    assert cli._resolve_port(argparse.Namespace(port=None), "codex") == 8398
    assert cli._resolve_port(argparse.Namespace(port=None), None) == 8257


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

    monkeypatch.setattr(daemon, "service_installed", lambda name=None: False)
    rc = cli.cmd_restart(argparse.Namespace(port=None, name=None))
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
