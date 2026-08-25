"""Daemon management: profile wiring and service definitions (no launchctl/systemctl)."""

import plistlib
import sys

from cliffcompaction import daemon


def test_env_block_posix():
    block = daemon.env_block(8399, fish=False)
    assert 'export ANTHROPIC_BASE_URL="http://127.0.0.1:8399"' in block
    assert 'export OPENAI_BASE_URL="http://127.0.0.1:8399/v1"' in block
    assert block.startswith(daemon.MARK_BEGIN)
    assert block.rstrip().endswith(daemon.MARK_END)


def test_env_block_fish():
    block = daemon.env_block(9000, fish=True)
    assert 'set -gx ANTHROPIC_BASE_URL "http://127.0.0.1:9000"' in block


def test_wire_profile_idempotent(tmp_path):
    profile = tmp_path / ".zshrc"
    profile.write_text("# my rc\nexport FOO=1\n")
    daemon.wire_profile(profile, 8399)
    daemon.wire_profile(profile, 9001)  # refresh with a new port
    text = profile.read_text()
    assert text.count(daemon.MARK_BEGIN) == 1
    assert "9001" in text and "8399" not in text
    assert "export FOO=1" in text  # untouched user content
    assert daemon.profile_is_wired(profile)


def test_unwire_profile_clean(tmp_path):
    profile = tmp_path / ".zshrc"
    profile.write_text("export FOO=1\n")
    daemon.wire_profile(profile, 8399)
    daemon.unwire_profile(profile)
    assert profile.read_text() == "export FOO=1\n"
    assert not daemon.profile_is_wired(profile)


def test_fish_profile_is_own_file(tmp_path):
    profile = tmp_path / "conf.d" / "cliffcompaction.fish"
    daemon.wire_profile(profile, 8399)
    assert profile.exists()
    assert "set -gx" in profile.read_text()
    daemon.unwire_profile(profile)
    assert not profile.exists()


def test_plist_content():
    data = plistlib.loads(daemon.plist_content(8399, ["--shadow"]))
    assert data["Label"] == daemon.LABEL
    assert data["ProgramArguments"][0] == sys.executable
    assert "-m" in data["ProgramArguments"]
    assert "cliffcompaction.cli" in data["ProgramArguments"]
    assert "serve" in data["ProgramArguments"]
    assert "8399" in data["ProgramArguments"]
    assert "--shadow" in data["ProgramArguments"]
    assert data["KeepAlive"] is True
    assert data["RunAtLoad"] is True


def test_systemd_unit_content():
    unit = daemon.systemd_unit_content(8399, [])
    assert f"ExecStart={sys.executable} -m cliffcompaction.cli serve --port 8399" in unit
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
