# CliffCompaction

A transparent API proxy implementing CliffCompaction.

## Quick start

```bash
uv tool install cliffcompaction   # or: pip install cliffcompaction

cliff enable
```

- `cliff enable` — installs the proxy as a supervised user service (launchd on macOS, systemd --user on Linux — auto-restarts, survives reboots) and wires `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL` into your shell profile. Open a new terminal for the env vars to take effect.
- `cliff status` — shows health.
- `cliff disable` — reverses everything.

For scoped, one-shot use (benchmarks, CI, trying it out):

```bash
# proxy up on a free port, env set for this command only, torn down after:
cliff run -- claude
cliff run -- python my_agent.py
```

Manual mode (flags optional — see Configuration):

```bash
cliff serve --port 8257 --threshold 128000 --keep-recent 3
export ANTHROPIC_BASE_URL=http://127.0.0.1:8257
export OPENAI_BASE_URL=http://127.0.0.1:8257/v1
```

**Important:** disable your scaffold's own compaction/summarization if it has one. With the proxy active your scaffold sees small prompt token counts, so its native triggers generally won't fire anyway — but disabling it is still recommended, since some native compactions rewrite history in place, which breaks the prefix matching CliffCompaction relies on.

## How it works

Every request an agent sends is `history + latest step`. The proxy:

1. Canonicalizes and hashes each message, forming a hash chain.
2. If the request's outgoing size exceeds the threshold (chars/4 estimate), it rebuilds the history as `[head verbatim] + [one CliffCompaction summary message] + [last keep_recent turns verbatim]` and stores the result keyed by the original prefix's chain hash.
3. Future requests arrive with the original (uncompacted) history. The proxy finds the longest stored prefix and substitutes the compacted version, forwarding `C + tail` instead of `S + tail`.
4. If the provider still rejects with a context-length error, the proxy compacts and replays once (reactive fallback).

The summary is mechanical, built by content class:

| Content | Treatment |
|---|---|
| Tool results | Kept verbatim iff ≤ 500 chars, dropped otherwise |
| Tool calls | One-line signatures (name + truncated arguments) |
| Assistant text & thinking | Kept in full by default (cap with `--thought-max-chars` / `--thinking-max-chars`; `--drop-thinking` excludes thinking entirely) |
| Human text | Verbatim in the current summary |
| Head (system + task) & recent turns | Untouched |

Re-compaction **drops** the previous summary.

**Fail-open contract:** any failure — unparseable body, no prefix match, store error — means verbatim passthrough.

## Shadow mode

```bash
cliff run --shadow -- claude     # or: cliff enable --shadow
```

Runs the full pipeline (hash, match, would-compact, log) but forwards every request verbatim. Use it to verify a scaffold's history is prefix-stable before going active, and to see what compaction would have saved.

## Configuration

| Flag / env var | Default | Meaning |
|---|---|---|
| `--threshold` / `CLIFF_THRESHOLD_TOKENS` | 128000 | proactive compaction trigger (est. tokens) |
| `--keep-recent` / `CLIFF_KEEP_RECENT` | 3 | recent assistant-step turns kept verbatim |
| `--thought-max-chars` / `CLIFF_THOUGHT_MAX_CHARS` | 0 (unlimited) | cap on assistant text in summaries |
| `--thinking-max-chars` / `CLIFF_THINKING_MAX_CHARS` | 0 (unlimited) | cap on thinking text in summaries, independent of the thought cap |
| `--drop-thinking` / `CLIFF_KEEP_THINKING=0` | keep | exclude thinking/reasoning text from summaries |
| `--result-max-chars` / `CLIFF_RESULT_MAX_CHARS` | 500 | tool results longer than this are dropped |
| `CLIFF_HUMAN_MAX_CHARS` | 20000 | sanity cap on human text in summaries |
| `--anthropic-upstream` / `CLIFF_ANTHROPIC_UPSTREAM` | `https://api.anthropic.com` | |
| `--openai-upstream` / `CLIFF_OPENAI_UPSTREAM` | `https://api.openai.com` | |
| `--shadow` / `CLIFF_SHADOW` | off | observe-only mode |
| `--debug-dir` / `CLIFF_DEBUG_DIR` | off | dump each handled request's incoming/outgoing message arrays as JSON files |

Supported dialects: **Anthropic Messages** (`/v1/messages`) and **OpenAI Chat Completions** (`/chat/completions`), native tool calling. Everything else passes through verbatim. Paths that match neither dialect (e.g. `/v1/models`) are forwarded to the Anthropic upstream — or to the OpenAI upstream when it is the only one you configured, so a single-provider OpenAI setup needs no extra flags.

## Development

```bash
uv sync
uv run pytest
```
