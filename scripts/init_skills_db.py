"""Create the skills database with all required tables.

Run once::

    uv run python scripts/init_skills_db.py

Creates ``./skills.db`` in the project root. Not part of the evolution
pipeline — not toggled by ``DRY_RUN``.
"""

import sqlite3
from pathlib import Path

SKILLS_DB_PATH = Path("skills.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS rules (
    id TEXT PRIMARY KEY,
    skill_name TEXT NOT NULL,
    content TEXT NOT NULL,
    helpful_count INTEGER DEFAULT 0,
    harmful_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rules_skill ON rules(skill_name);

CREATE TABLE IF NOT EXISTS processed_synthesize (
    session_id TEXT PRIMARY KEY,
    processed_at TEXT NOT NULL,
    skill_name TEXT,
    action TEXT
);

CREATE TABLE IF NOT EXISTS processed_evolve (
    session_id TEXT PRIMARY KEY,
    processed_at TEXT NOT NULL,
    rules_tagged INTEGER DEFAULT 0,
    rules_added INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS skill_clusters (
    skill_name TEXT NOT NULL,
    cluster_id INTEGER NOT NULL,
    goal_text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (skill_name, cluster_id, goal_text)
);
"""


def main() -> None:
    db = SKILLS_DB_PATH
    exists = db.exists()
    conn = sqlite3.connect(db)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    tag = "Re-initialized" if exists else "Created"
    print(f"{tag} skills database at {db.resolve()}")


if __name__ == "__main__":
    main()
