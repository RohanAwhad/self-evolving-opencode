# AGENTS.md

## Quick start

```bash
uv sync                          # install deps into .venv
uv run python play.py            # list OpenCode sessions
uv run python play.py --goals SESSION_ID          # extract goals from a session
uv run python play.py --goals SESSION_ID --check   # also check if goals were achieved
```

Redis (optional — used for LLM call caching, degrades gracefully if down):
```bash
docker compose up -d             # starts seo-redis on localhost:6380
```

## Project overview

CLI pipeline that reads OpenCode's SQLite DB (`~/.local/share/opencode/opencode.db`), extracts user goals from conversation transcripts via LLM, checks if they were achieved, and clusters goals by similarity.

## Architecture

- **`play.py`** — CLI entrypoint (imports from `src/`). `main.py` is a stub — ignore it.
- **`src/opencode_db.py`** — all SQLite access: sessions, messages, transcripts, message slicing. Default DB path: `~/.local/share/opencode/opencode.db`.
- **`src/llm/__init__.py`** — thin wrapper around `AnthropicVertex`. All LLM calls go through `complete()` (text) or `complete_tool()` (forced tool-use). Both cache results in Redis.
- **`src/llm/cache.py`** — Redis-backed LLM response cache. Connects to `localhost:6380`. Silently degrades if Redis is unavailable.
- **`src/goal_extractor.py`** — extracts goals from a session transcript via forced tool-use.
- **`src/goal_checker.py`** — checks if a goal was achieved via forced tool-use. Returns `GoalResult(achieved, reasoning)`.
- **`src/goal_clusterer.py`** — clusters goal strings with `sentence-transformers` embeddings + HDBSCAN. Has its own CLI: `uv run python -m src.goal_clusterer`.

## Conventions

- **Configurable defaults** (see `CLAUDE.md`): every tunable parameter must be a function/constructor parameter with a sensible default. Never hardcode inline.
- LLM model default: `claude-opus-4-6@default` (Vertex AI, not direct Anthropic API).
- Requires GCP auth: `gcloud auth application-default login`.
- Env vars (defaults usually set in zshrc): `ANTHROPIC_VERTEX_PROJECT_ID` (`itpc-gcp-ai-eng-claude`), `GOOGLE_VERTEX_LOCATION` (`global`).
- Redis env vars: `REDIS_HOST` (`localhost`), `REDIS_PORT` (`6380`).
- Python 3.12, managed by `uv`. No pip, no poetry.

## Current state

- No tests, no CI, no linting/formatting config.
