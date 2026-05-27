# AGENTS.md

## Quick start

```bash
uv sync                          # install deps into .venv
uv run python play.py            # list OpenCode sessions
uv run python play.py --goals SID          # extract goals from a session
uv run python play.py --goals SID --check  # check if goals achieved
uv run python play.py --goals SID --summarize  # summarize a thread
uv run python play.py --evolve [N]         # skill evolution pipeline (default 50)
```

Filters: `--dir SUBSTRING`, `--agent NAME`, `-n LIMIT`. `--goals` accepts row indices too.

Batch: `--goals-file PATH` (one session ID per line). Clustering: `--goals-file PATH --cluster`.

## Testing

```bash
uv run pytest                    # all except @live (default addopts)
uv run pytest -m live            # hits real LLM — needs GCP auth
uv run pytest -m redis           # only Redis-dependent tests
uv run pytest -k test_opencode   # single test file
uv run pytest -k TestGetSessions # single test class
```

No mocks. Tests use real SQLite fixtures, real Redis, pre-seeded LLM cache hits. `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed.

Fixtures from `tests/conftest.py`:
- `db_path` — tmp SQLite with 5 sessions/13 messages/15 parts (includes tool:skill parts)
- `skills_db_path` — tmp skills.db with all Phase 2 tables
- `temp_skills_dir` — empty dir for scan_skills tests
- `redis_client` — yields `(r, seeded_keys)`, auto-cleans up
- `_reset_llm_redis` — autouse, fixes event-loop mismatch between test runs

Cache pre-seeding (`tests/helpers.py`): `preseed_complete()`, `preseed_complete_tool()`.

Smoke tests: `uv run pytest tests/test_skill_evolution.py -m live --override-ini="addopts="`

## Architecture

**Phase 1** (built): `play.py` → `opencode_db.py` → `goal_extractor.py` → `goal_checker.py` → `conversation_summarizer.py` → `goal_clusterer.py`

**Phase 2** (built): Two sequential queues via `--evolve`:
1. **Synthesizer** (oldest-first): extract goals → cluster → semantic search → LLM decide (new/update) → synthesize SKILL.md
2. **Evolve** (newest-first): detect skills (tool:skill parts) → reflect per thread → curate per skill → append rules to SKILL.md

Pipeline orchestrator: `src/skill_evolution.py` — `run_evolve()`.

Key modules:
- `src/skill_rules.py` — SQLite rule DB: RuleRow, RuleTag, RuleStats, CRUD, counter updates, `next_rule_ids()`
- `src/skill_registry.py` — scan SKILL.md files, semantic search (description embeddings), LLM decide new/update, dual-queue session tracking
- `src/skill_synthesizer.py` — LLM generates SKILL.md from cluster goals + thread summaries. Two modes: new (full) and update (refine workflow, preserve rules)
- `src/reflector.py` — tags rules (4 tags: irrelevant/followed_helpful/followed_harmful/not_followed), extracts insights grouped by skill
- `src/curator.py` — ADD-only rule synthesis from reflector insights. Deduplicates via cosine similarity (threshold 0.90)

All LLM calls go through `src/llm/__init__.py` (`complete()` for text, `complete_tool()` for forced tool-use), cached in Redis, auto-retry 5x with exponential backoff.

Specs: `specs/000_overview.md` (roadmap), `specs/001-013` (per-module).

## DBs

OpenCode DB: `~/.local/share/opencode/opencode.db` (read-only, sessions/messages/parts).

Skills DB: `./skills.db` (project root). Initialize once:
```bash
uv run python scripts/init_skills_db.py
```
Tables: `rules`, `processed_synthesize`, `processed_evolve`, `skill_clusters`.

Skills write to `~/.claude/skills/<name>/SKILL.md`.

## DRY_RUN

```bash
DRY_RUN=1 uv run python play.py --evolve 5
```
Runs all LLM calls, prints "would write" messages, zero disk/SQLite writes.

## Gotchas

- Redis on `localhost:6380`, not default 6379. `podman compose up -d` (Podman, not Docker).
- LLM: `claude-opus-4-6@default` via Vertex AI. Requires `gcloud auth application-default login`. Env vars: `ANTHROPIC_VERTEX_PROJECT_ID`, `GOOGLE_VERTEX_LOCATION`.
- `loguru` is the logger. No configuration needed — goes to stderr by default. Log level: `LOGGING_LEVEL` env var.
- The `_redis` client in `src/llm/__init__.py` is a module-level singleton. When running multiple `asyncio.run()` calls in the same process (e.g. multi-scenario scripts), reset it between runs: `src.llm._redis = aioredis.Redis(...)`.
- `pytest_collection_modifyitems` skips `@pytest.mark.live` tests unless `-m live` is explicitly passed. The `addopts` config reads `-m 'not live'` — use `--override-ini="addopts="` to run live tests.

## Conventions

- Every function parameter has a sensible default. Never hardcode inline.
- All DB functions accept `db_path: Path` kwarg for testability.
- No `try/except` unless explicit. Let errors surface.
- `asyncio.to_thread()` wraps all sync SQLite operations in async functions.
- Rule IDs: `skillname-00001` format, five-digit zero-padded. Counters live in SQLite only, never in markdown.
- Curator is ADD-only — never modifies or deletes existing rules.
- Python 3.12, managed by `uv`. No pip, no poetry. No CI, no linting config.
