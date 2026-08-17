"""cliff CLI.

    cliff enable [options]                     # daily driver: supervised daemon + shell env wiring
    cliff disable                              # remove the daemon and env wiring
    cliff status                               # daemon / env / store health
    cliff serve [--shadow] [--port N] ...      # run the proxy in the foreground
    cliff run [--shadow] [options] -- CMD ...  # wrap one command: proxy up, env set, run, tear down

`cliff enable` and `cliff run` set ANTHROPIC_BASE_URL and OPENAI_BASE_URL so
any scaffold talks through the proxy with zero integration code.
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import subprocess
import sys
import threading
import time

from . import __version__
from .config import Config
from .proxy import create_app

logger = logging.getLogger("cliffcompaction")


def _setup_logging(verbose: bool) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s cliff %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)


def _add_common_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--shadow", action="store_true", help="observe and log, never modify a request")
    p.add_argument("--threshold", type=int, help="proactive compaction threshold in est. tokens (default 100000)")
    p.add_argument("--keep-recent", type=int, help="recent turns kept verbatim (default 3)")
    p.add_argument("--thought-max-chars", type=int, help="cap on assistant text per summarized turn; 0 = unlimited (default)")
    p.add_argument("--result-max-chars", type=int, help="tool results longer than this are dropped (default 500)")
    p.add_argument("--drop-thinking", action="store_true", help="exclude thinking/reasoning text from summaries (leaner configuration)")
    p.add_argument("--thinking-max-chars", type=int, help="cap on thinking text per summarized turn; 0 = unlimited (default), independent of --thought-max-chars")
    p.add_argument("--anthropic-upstream", help="Anthropic upstream base URL")
    p.add_argument("--openai-upstream", help="OpenAI-compatible upstream base URL")
    p.add_argument("-v", "--verbose", action="store_true")


def _config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config.from_env()
    if args.shadow:
        cfg.shadow = True
    if args.threshold is not None:
        cfg.threshold_tokens = args.threshold
    if args.keep_recent is not None:
        cfg.keep_recent = args.keep_recent
    if args.thought_max_chars is not None:
        cfg.thought_max_chars = args.thought_max_chars
    if args.result_max_chars is not None:
        cfg.result_max_chars = args.result_max_chars
    if args.drop_thinking:
        cfg.keep_thinking = False
    if args.thinking_max_chars is not None:
        cfg.thinking_max_chars = args.thinking_max_chars
    if args.anthropic_upstream:
        cfg.anthropic_upstream = args.anthropic_upstream
    if args.openai_upstream:
        cfg.openai_upstream = args.openai_upstream
    return cfg


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server_thread(cfg: Config) -> "uvicorn.Server":
    import uvicorn

    app = create_app(cfg)
    uv_cfg = uvicorn.Config(app, host=cfg.host, port=cfg.port, log_level="warning")
    server = uvicorn.Server(uv_cfg)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("proxy failed to start within 10s")
        time.sleep(0.02)
    return server


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    cfg = _config_from_args(args)
    if args.port is not None:
        cfg.port = args.port
    if args.host:
        cfg.host = args.host
    mode = "shadow" if cfg.shadow else "active"
    logger.info(
        "cliffcompaction %s serving on http://%s:%d (%s mode, threshold ~%dk tokens, keep_recent=%d)",
        __version__, cfg.host, cfg.port, mode, cfg.threshold_tokens // 1000, cfg.keep_recent,
    )
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level="warning")
    return 0


def cmd_run(args: argparse.Namespace, command: list[str]) -> int:
    if not command:
        print("usage: cliff run [options] -- COMMAND [ARGS...]", file=sys.stderr)
        return 2
    cfg = _config_from_args(args)
    cfg.port = _free_port()
    server = _start_server_thread(cfg)
    base = f"http://{cfg.host}:{cfg.port}"
    env = dict(os.environ)
    env["ANTHROPIC_BASE_URL"] = base
    env["OPENAI_BASE_URL"] = base + "/v1"
    env["OPENAI_API_BASE"] = base + "/v1"  # legacy SDKs
    mode = "shadow" if cfg.shadow else "active"
    logger.info("proxy on %s (%s mode); running: %s", base, mode, " ".join(command))
    try:
        proc = subprocess.run(command, env=env)
        return proc.returncode
    finally:
        server.should_exit = True


def _serve_args_from(args: argparse.Namespace) -> list[str]:
    """Forward explicitly-set common flags to the daemon's serve command."""
    out: list[str] = []
    if args.shadow:
        out.append("--shadow")
    if args.threshold is not None:
        out += ["--threshold", str(args.threshold)]
    if args.keep_recent is not None:
        out += ["--keep-recent", str(args.keep_recent)]
    if args.thought_max_chars is not None:
        out += ["--thought-max-chars", str(args.thought_max_chars)]
    if args.result_max_chars is not None:
        out += ["--result-max-chars", str(args.result_max_chars)]
    if args.drop_thinking:
        out.append("--drop-thinking")
    if args.thinking_max_chars is not None:
        out += ["--thinking-max-chars", str(args.thinking_max_chars)]
    if args.anthropic_upstream:
        out += ["--anthropic-upstream", args.anthropic_upstream]
    if args.openai_upstream:
        out += ["--openai-upstream", args.openai_upstream]
    if args.verbose:
        out.append("-v")
    return out


def cmd_enable(args: argparse.Namespace) -> int:
    from pathlib import Path

    from . import daemon

    port = args.port if args.port is not None else Config.from_env().port
    try:
        daemon.start_service(port, _serve_args_from(args))
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    status = daemon.wait_healthy(port)
    if status is None:
        print(
            f"error: daemon installed but not responding on port {port}; "
            f"check {daemon.log_path()}",
            file=sys.stderr,
        )
        return 1
    profile = Path(args.profile) if args.profile else daemon.default_profile()
    if not args.no_env:
        daemon.wire_profile(profile, port)
    mode = "shadow" if status.get("shadow") else "active"
    print(f"cliffcompaction enabled: proxy on http://127.0.0.1:{port} ({mode} mode)")
    print(f"  supervised: auto-restarts, survives reboots (logs: {daemon.log_path()})")
    if args.no_env:
        print("  env wiring skipped (--no-env); set ANTHROPIC_BASE_URL/OPENAI_BASE_URL yourself")
    else:
        print(f"  env wired in {profile}")
        print("  open a new terminal (or `source` your profile) for agents to pick it up")
    return 0


def cmd_disable(args: argparse.Namespace) -> int:
    from pathlib import Path

    from . import daemon

    daemon.stop_service()
    profile = Path(args.profile) if args.profile else daemon.default_profile()
    daemon.unwire_profile(profile)
    print("cliffcompaction disabled: daemon removed, env wiring removed")
    print("  open a new terminal for the env change to take effect")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from . import daemon

    port = args.port if args.port is not None else Config.from_env().port
    installed = daemon.service_installed()
    status = daemon.probe(port)
    wired = daemon.profile_is_wired(daemon.default_profile())
    print(f"daemon installed : {'yes' if installed else 'no'}")
    if status is not None:
        mode = "shadow" if status.get("shadow") else "active"
        print(
            f"proxy responding : yes on port {port} ({mode} mode, "
            f"threshold ~{status.get('threshold_tokens', 0) // 1000}k tokens, "
            f"keep_recent={status.get('keep_recent')}, "
            f"store entries={status.get('store_entries')})"
        )
    else:
        print(f"proxy responding : no (port {port})")
    print(f"env wired        : {'yes' if wired else 'no'} ({daemon.default_profile()})")
    if installed and status is None:
        print(f"hint: check the log at {daemon.log_path()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Split off the wrapped command for `run` before argparse sees it.
    command: list[str] = []
    if "--" in argv:
        idx = argv.index("--")
        command = argv[idx + 1 :]
        argv = argv[:idx]

    parser = argparse.ArgumentParser(prog="cliff", description=__doc__)
    parser.add_argument("--version", action="version", version=f"cliffcompaction {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="run the proxy in the foreground")
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.add_argument("--host", default=None)
    _add_common_flags(p_serve)

    p_run = sub.add_parser("run", help="wrap a command behind the proxy")
    _add_common_flags(p_run)

    p_enable = sub.add_parser("enable", help="install the supervised daemon + shell env wiring")
    p_enable.add_argument("--port", type=int, default=None)
    p_enable.add_argument("--profile", help="shell profile file to wire (default: auto-detect)")
    p_enable.add_argument("--no-env", action="store_true", help="install the daemon but skip shell env wiring")
    _add_common_flags(p_enable)

    p_disable = sub.add_parser("disable", help="remove the daemon and env wiring")
    p_disable.add_argument("--profile", help="shell profile file to unwire (default: auto-detect)")

    p_status = sub.add_parser("status", help="daemon / env / store health")
    p_status.add_argument("--port", type=int, default=None)

    args = parser.parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))

    if args.cmd == "serve":
        return cmd_serve(args)
    if args.cmd == "run":
        return cmd_run(args, command)
    if args.cmd == "enable":
        return cmd_enable(args)
    if args.cmd == "disable":
        return cmd_disable(args)
    if args.cmd == "status":
        return cmd_status(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
