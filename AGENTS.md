# AGENTS.md

## Quick start

```bash
uv sync                          # install deps into .venv
uv run python play.py            # list OpenCode sessions
uv run python play.py --goals SESSION_ID          # extract goals from a session
uv run python play.py --goals SESSION_ID --check   # also check if goals were achieved
```

Filters: `--dir SUBSTRING`, `--agent NAME`, `-n LIMIT`. Combine freely.

`--goals` accepts a **row index** (from the listing) as well as a session ID — e.g. `--goals 3` picks the 3rd row.

Batch mode: `--goals-file PATH` reads one session ID per line (see `sessions.txt` for an example).

Clustering CLI (separate entrypoint): `uv run python -m src.goal_clusterer`.

Testing:
```bash
uv run pytest                    # runs all tests except @live (default addopts)
uv run pytest -m live            # hits real LLM API (slow, needs GCP auth)
uv run pytest -m redis           # only Redis-dependent tests
uv run pytest -k test_opencode   # run a single test file
```

Redis (optional — used for LLM call caching, degrades gracefully if down):
```bash
docker compose up -d             # starts seo-redis on localhost:6380 (not default 6379)
```

## Project overview

CLI pipeline that reads OpenCode's SQLite DB (`~/.local/share/opencode/opencode.db`), extracts user goals from conversation transcripts via LLM, checks if they were achieved, and clusters goals by similarity.

## Architecture

- **`play.py`** — CLI entrypoint. `main.py` is a stub — ignore it.
- **`src/opencode_db.py`** — all SQLite access: sessions, messages, transcripts, message slicing. DB path: `~/.local/share/opencode/opencode.db`.
- **`src/llm/__init__.py`** — thin wrapper around `AsyncAnthropicVertex`. All LLM calls go through `complete()` (text) or `complete_tool()` (forced tool-use). Both cache in Redis. Auto-retries 5 times with exponential backoff (tenacity).
- **`src/llm/cache.py`** — Redis cache on `localhost:6380`. Silently degrades if Redis is unavailable.
- **`src/goal_extractor.py`** — extracts goals via forced tool-use.
- **`src/goal_checker.py`** — checks if a goal was achieved. Returns `GoalResult(achieved, reasoning)`.
- **`src/goal_clusterer.py`** — clusters goal strings with `sentence-transformers` embeddings + HDBSCAN. Has its own CLI: `uv run python -m src.goal_clusterer`.
- **`src/conversation_summarizer.py`** — summarizes rich messages into structured markdown (Goal/Intent/Outcome sections). Uses delimiter tags to extract output.

Everything is **async** — entry via `asyncio.run(main())`.

## Testing

- **No mocks.** Tests use real SQLite fixture DBs (seeded via `conftest.py`), real Redis, pre-recorded LLM responses in `tests/fixtures/`.
- `conftest.py` provides: `db_path` fixture (tmp SQLite with 5 sessions/11 messages/12 parts), `redis_client` fixture (skips if Redis down), `_reset_llm_redis` (autouse, prevents event-loop mismatch).
- `tests/helpers.py` has `preseed_complete_tool()` for pre-seeding Redis cache with LLM responses.
- All DB functions accept a `db_path` kwarg — tests pass the fixture path, production uses `~/.local/share/opencode/opencode.db`.
- `asyncio_mode = "auto"` in pyproject.toml — no need for `@pytest.mark.asyncio` on tests.
- Test plan details in `.test_plans.md`.

## Conventions

- **Configurable defaults**: every tunable parameter must be a function/constructor parameter with a sensible default. Never hardcode inline.
- LLM model: `claude-opus-4-6@default` via **Vertex AI** (not direct Anthropic API).
- Requires GCP auth: `gcloud auth application-default login`.
- Env vars (defaults usually set in zshrc): `ANTHROPIC_VERTEX_PROJECT_ID` (`itpc-gcp-ai-eng-claude`), `GOOGLE_VERTEX_LOCATION` (`global`).
- Redis env vars: `REDIS_HOST` (`localhost`), `REDIS_PORT` (`6380`).
- Python 3.12, managed by `uv`. No pip, no poetry. No CI, no linting/formatting config.
- Container runtime: **Podman** (`podman compose`), not Docker.
