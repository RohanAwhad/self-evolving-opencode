# self-evolving-opencode

CLI tool that reads OpenCode's SQLite DB, lists conversation sessions, extracts user goals via LLM, and checks if they were achieved.

## Setup

```bash
uv init    # first time only
uv sync
```

Requires GCP auth for Vertex AI:
```bash
gcloud auth application-default login
```

Env vars (defaults work if your zshrc sets them):
- `ANTHROPIC_VERTEX_PROJECT_ID` (default: `itpc-gcp-ai-eng-claude`)
- `GOOGLE_VERTEX_LOCATION` (default: `global`)

## Usage

### List sessions

```bash
# most recent 30 sessions
uv run python play.py

# limit to 10
uv run python play.py -n 10

# filter by agent
uv run python play.py --agent auto-accept

# filter by directory (substring match)
uv run python play.py --dir my-project

# combine filters
uv run python play.py --agent auto-accept --dir my-project -n 5
```

### Extract goals from a session

```bash
# by session ID
uv run python play.py --goals ses_abc123...

# by row index from listing (e.g. row 6 from --agent filter)
uv run python play.py --agent auto-accept --goals 6
```

### Check if goals were achieved

```bash
# extract goals + check each against conversation evidence
uv run python play.py --goals ses_abc123... --check

# with index + filter
uv run python play.py --agent auto-accept --goals 6 --check
```

## How it works

1. **List mode** -- queries the `session` table in `~/.local/share/opencode/opencode.db`, joins with `message` for counts
2. **Goals mode** (`--goals`) -- loads the full conversation transcript (messages + parts) and sends it to Claude Opus via Vertex AI, which identifies distinct goals/threads using forced tool use
3. **Check mode** (`--check`) -- for each extracted goal, slices the relevant messages and calls `src/goal_checker.py` to evaluate whether the goal was achieved, again via forced tool use
