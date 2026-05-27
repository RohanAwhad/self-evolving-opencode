# Devlogs

## 2025-05-26 — Phase 2 Implementation

### Completed: Full skill evolution pipeline

Implemented all Phase 2 modules (specs 008-013). 176 tests total.

**Module order (easiest → hardest):**

1. **`src/skill_rules.py`** — SQLite rule DB with RuleRow, RuleTag, RuleStats dataclasses. CRUD operations: insert_rules, get_rules_for_skill, update_counters, get_max_rule_id, get_rule_stats, next_rule_ids. 18 tests.

2. **`src/opencode_db.py`** — Added `get_skills_for_session()` to query tool:skill parts. Extended seed data in conftest.py with skill invocation parts. 35 tests (30 existing + 5 new).

3. **`scripts/init_skills_db.py`** — Creates all Phase 2 tables in `./skills.db`: rules, processed_synthesize, processed_evolve, skill_clusters. Added to .gitignore.

4. **`src/skill_registry.py`** — Skill scanning (YAML frontmatter parsing), semantic search (description embeddings + cosine similarity), LLM-based new-vs-update decision, dual-queue session tracking (synthesize=oldest first, evolve=newest first). 26 tests.

5. **`src/skill_synthesizer.py`** — LLM-based skill creation/update from cluster goals + thread summaries. Two modes: new (full SKILL.md) and update (preserve rules, refine workflow). 2 tests with pre-seeded cache.

6. **`src/reflector.py`** — Per-thread rule tagging (irrelevant/followed_helpful/followed_harmful/not_followed) and insight extraction grouped by skill. Two modes: tag mode (with rules) and insight-only (no rules). 3 tests with pre-seeded cache.

7. **`src/curator.py`** — Per-skill ADD-only rule synthesis from reflector insights. Programmatic dedup via cosine similarity (threshold 0.90). 3 tests with pre-seeded cache.

8. **`src/skill_evolution.py`** — Main pipeline orchestrator. Two sequential queues: synthesizer (oldest-first, extracts goals→clusters→synthesizes skills) and evolve (newest-first, reflects→curates→writes rules). DRY_RUN=1 prevents all disk/DB writes.

9. **`play.py`** — Added `--evolve [N]` CLI flag. Defaults to 50 sessions per queue.

10. **Smoke tests** (`tests/test_skill_evolution.py`) — 4 scenarios: synthesize-bootstrap (new skill), synthesize-existing (update workflow), evolve-bootstrap (insight-only, no rules), evolve-existing (tag + curate). All pre-seeded Redis cache. 5 tests marked `@pytest.mark.live`.

**Test totals**: 171 unit/integration (default) + 5 smoke (`-m live`) = 176

**Design decisions finalized during implementation:**
- `_next_rule_id` adds +1 to max_id (consistent naming), `_next_rule_ids_sync` passes `max_id + i` (not double-counting)
- Empty YAML frontmatter (`---\n---`) returns None, not {}
- Confirmed `pytest_collection_modifyitems` only skips live tests when `-m live` not passed
- Reflector's `reflect_insight_only` takes explicit `skill_names` param (spec doesn't show it but it's necessary)

### Remaining
- Run live smoke tests against real OpenCode DB with a few sessions
- Profile pipeline performance, consider asyncio.gather for parallel queue execution
- Consider adding `--force` flag to re-process sessions
- `_get_unprocessed_sessions_sync` uses `NOT IN (...)` which won't scale to thousands of processed sessions — switch to LEFT JOIN or temp table
