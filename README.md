# self-evolving-opencode

CLI pipeline that reads OpenCode's SQLite DB, extracts user goals via LLM, clusters them, and automatically synthesizes + evolves Claude skills from conversation patterns.

## Setup

```bash
uv sync
```

Requires GCP auth for Vertex AI:
```bash
gcloud auth application-default login
```

Redis (for LLM call caching — degrades gracefully if down):
```bash
podman compose up -d    # starts seo-redis on localhost:6380
```

Initialize the skills database (once):
```bash
uv run python scripts/init_skills_db.py
```

Env vars (defaults work if your zshrc sets them):
- `ANTHROPIC_VERTEX_PROJECT_ID` (default: `itpc-gcp-ai-eng-claude`)
- `GOOGLE_VERTEX_LOCATION` (default: `global`)

## Skill Evolution

The main feature. Mines OpenCode conversations to create and improve Claude skills automatically.

```bash
# 1. Dry run — inspect what it would do (no writes to disk or DB)
DRY_RUN=1 uv run python play.py --evolve 5

# 2. Real run — writes to ~/.claude/skills/ and ./skills.db
uv run python play.py --evolve 5

# 3. Check what it wrote
ls ~/.claude/skills/
cat ~/.claude/skills/*/SKILL.md
```

`--evolve [N]` runs two sequential queues (default N=50 sessions per queue):

1. **Synthesizer** (oldest-first): extract goals → cluster → semantic search existing skills → LLM decide new/update → synthesize SKILL.md
2. **Evolve** (newest-first): detect skill invocations → reflect per thread (tag rules + extract insights) → curate per skill (ADD new rules)

Options:
- `--concurrency M` — max concurrent LLM calls (default: 5)
- `DRY_RUN=1` — runs all LLM calls, prints SKILL.md content to stdout, zero disk/DB writes

Skills are written to `~/.claude/skills/<name>/SKILL.md`. Rules are tracked in `./skills.db`.

## Usage

### List sessions

```bash
uv run python play.py                              # most recent 30
uv run python play.py -n 10                         # limit
uv run python play.py --agent auto-accept           # filter by agent
uv run python play.py --dir my-project              # filter by directory
uv run python play.py --agent auto-accept --dir my-project -n 5
```

### Extract goals

```bash
uv run python play.py --goals ses_abc123...         # by session ID
uv run python play.py --agent auto-accept --goals 6 # by row index
uv run python play.py --goals ses_abc123... --check  # check if achieved
uv run python play.py --goals ses_abc123... --summarize  # summarize thread
```

### Batch + clustering

```bash
uv run python play.py --goals-file sessions.txt              # batch extract
uv run python play.py --goals-file sessions.txt --cluster     # cluster goals
uv run python play.py --goals-file sessions.txt --cluster --summarize  # + summaries
```

## Testing

```bash
uv run pytest                    # all except @live (default)
uv run pytest -m live --override-ini="addopts="   # hits real LLM API
uv run pytest -m redis           # only Redis-dependent tests
uv run pytest -k test_opencode   # single test file
```

## How it works

1. **List mode** — queries `session` table in `~/.local/share/opencode/opencode.db`
2. **Goals mode** (`--goals`) — sends conversation transcript to Claude Opus via Vertex AI, extracts goals via forced tool use
3. **Check mode** (`--check`) — evaluates whether each goal was achieved
4. **Evolve mode** (`--evolve`) — full pipeline: extract goals → cluster → synthesize skills → reflect on threads → curate rules
