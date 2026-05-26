# 000 — Architecture Overview

## Purpose

CLI pipeline that reads OpenCode's SQLite database, extracts user goals from conversation transcripts via LLM (Claude Opus via Vertex AI), checks whether goals were achieved, and clusters goals by semantic similarity.

## Module Map

```
play.py (CLI entrypoint)
    │
    ├─► opencode_db.py        SQLite access for sessions, messages, transcripts
    │
    ├─► goal_extractor.py     LLM extracts goals from conversation transcripts
    │       └─► llm/
    │            ├─ complete()        text completions
    │            ├─ complete_tool()   forced tool-use completions
    │            └─ cache.py          Redis caching (silent degradation)
    │
    ├─► goal_checker.py       LLM checks if a goal was achieved
    │       └─► llm/
    │
    ├─► conversation_summarizer.py   LLM summarizes tool-use messages → markdown
    │       └─► llm/
    │
    └─► goal_clusterer.py     Embedding + agglomerative clustering of goal strings
            (sentence-transformers, scipy, sklearn — no LLM)
```

## Data Flow

```
OpenCode SQLite DB (~/.local/share/opencode/opencode.db)
  │
  ├─ get_sessions()           → list[Session]           (listing mode)
  │
  ├─ extract_goals(sid)       → list[Goal]              (--goals)
  │     └─ get_conversation_transcript(sid)
  │
  ├─ check_goal_achieved()    → GoalResult              (--goals --check)
  │     └─ get_messages_for_session(sid)
  │     └─ slice_messages(msgs, goal.message_range)
  │
  ├─ summarize_conversation() → markdown string         (--goals --summarize)
  │     └─ get_rich_messages_for_session(sid)
  │     └─ slice_messages(msgs, goal.message_range)
  │
  └─ cluster_goals(goal_texts) → ClusterResult           (--goals-file --cluster)
        (embeddings → mean-subtract → L2-norm → cosine distances → linkage → fcluster)
```

## Key Data Types

```python
Session(id, title, directory, agent, model_id, cost, tokens_input, tokens_output, time_created, time_updated, message_count)
Goal(title, description, message_range)
GoalResult(achieved: bool, reasoning: str)
ClusterResult(clusters: dict[int, list[str]], labels: list[int])
```

## External Dependencies

- **LLM**: Claude Opus 4 via Vertex AI (AsyncAnthropicVertex)
- **Redis**: localhost:6380 (Podman compose), optional — degrades silently
- **GCP auth**: `gcloud auth application-default login`
- **Sentence transformers**: `all-mpnet-base-v2` for embeddings

## Design Conventions

- Everything async, entry via `asyncio.run(main())`
- All DB functions accept `db_path` kwarg (testability)
- No mocks in tests — real SQLite fixtures, real Redis, pre-recorded LLM responses
- Configurable defaults on every function parameter, never hardcoded inline
- No try/except unless explicitly requested — let errors surface

## Testing

**Philosophy**: No mocks. All test data is real: SQLite fixture DBs, real Redis, pre-recorded LLM responses, real embeddings.

**Framework**: pytest + pytest-asyncio (`asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed)

**Run**:
```bash
uv run pytest                    # all except @live (default addopts)
uv run pytest -m live            # hits real LLM API (slow)
uv run pytest -m redis           # only Redis-dependent tests
```

**Fixtures** (from `tests/conftest.py`):
- `db_path` — tmp SQLite with schema + seed data (5 sessions, 11 messages, 12 parts)
- `redis_client` — async Redis on `localhost:6380`, skips if unavailable
- `_reset_llm_redis` — autouse, prevents event-loop mismatch

**Cache pre-seeding** (`tests/helpers.py`):
- `preseed_complete_tool()` — pre-seed Redis with LLM tool-use response
- `preseed_complete()` — pre-seed Redis with LLM text response
- Both use deterministic `cache_key()` to match cache lookup

**Test layout** (114 total tests across 6 files):

| File | Tests | Type | Dependencies |
|---|---|---|---|
| `test_opencode_db.py` | 30 | Unit | SQLite fixture only |
| `test_goal_clusterer.py` | 24 | Unit | Pure compute, no external deps |
| `test_llm.py` | 24 | Unit + Integration | Redis fixture (`@redis`) |
| `test_goal_checker.py` | 13 | Unit + Integration | Redis fixture (`@redis`) |
| `test_conversation_summarizer.py` | 19 | Unit + Integration | Redis fixture (`@redis`) |
| `test_goal_extractor.py` | 4 | Integration | DB + Redis fixtures (`@redis`) |

**Markers**: `@pytest.mark.redis` (needs Redis), `@pytest.mark.live` (hits real LLM, skipped by default via `conftest.py` autoskip)
