"""Tests for src/skill_registry.py."""

import sqlite3
from pathlib import Path

from src.skill_registry import (
    SkillInfo,
    _parse_frontmatter,
    get_unprocessed_sessions,
    is_session_processed,
    mark_sessions_processed,
    scan_skills,
    search_similar,
)


# -- _parse_frontmatter (pure) --------------------------------------------------


class TestParseFrontmatter:
    def test_valid_frontmatter(self):
        content = "---\nname: test-skill\ndescription: A test skill\n---\n\n## Workflow\n..."
        fm = _parse_frontmatter(content)
        assert fm == {"name": "test-skill", "description": "A test skill"}

    def test_no_frontmatter(self):
        content = "## Workflow\nSome content"
        assert _parse_frontmatter(content) is None

    def test_invalid_yaml(self):
        content = "---\n[\n---\n"
        assert _parse_frontmatter(content) is None

    def test_empty_frontmatter(self):
        content = "---\n---\n\nbody"
        fm = _parse_frontmatter(content)
        assert fm is None


# -- scan_skills (async, needs dir) ---------------------------------------------


class TestScanSkills:
    async def test_empty_dir(self, temp_skills_dir):
        skills = await scan_skills(temp_skills_dir)
        assert skills == []

    async def test_single_skill(self, temp_skills_dir):
        skill_dir = temp_skills_dir / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: A test skill\n---\n\n## Workflow\nDo stuff.\n\n## Rules\n- Rule 1\n"
        )
        skills = await scan_skills(temp_skills_dir)
        assert len(skills) == 1
        assert skills[0].name == "test-skill"
        assert skills[0].description == "A test skill"
        assert skills[0].has_rules is True

    async def test_skill_without_rules(self, temp_skills_dir):
        skill_dir = temp_skills_dir / "simple-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: simple-skill\ndescription: Simple\n---\n\n## Workflow\nJust do it.\n"
        )
        skills = await scan_skills(temp_skills_dir)
        assert len(skills) == 1
        assert skills[0].has_rules is False

    async def test_multiple_skills(self, temp_skills_dir):
        for name in ["skill-a", "skill-b", "skill-c"]:
            d = temp_skills_dir / name
            d.mkdir()
            (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Desc of {name}\n---\n\n## Workflow\n...\n")
        skills = await scan_skills(temp_skills_dir)
        assert len(skills) == 3
        assert sorted(s.name for s in skills) == ["skill-a", "skill-b", "skill-c"]

    async def test_skill_without_name_uses_dirname(self, temp_skills_dir):
        skill_dir = temp_skills_dir / "auto-name"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\ndescription: No name field\n---\n\nbody")
        skills = await scan_skills(temp_skills_dir)
        assert len(skills) == 1
        assert skills[0].name == "auto-name"

    async def test_dir_with_no_skill_md_skipped(self, temp_skills_dir):
        skill_dir = temp_skills_dir / "empty-skill"
        skill_dir.mkdir()
        skills = await scan_skills(temp_skills_dir)
        assert skills == []

    async def test_full_path_in_result(self, temp_skills_dir):
        skill_dir = temp_skills_dir / "path-test"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("---\nname: path-test\ndescription: Test\n---\n\nbody")
        skills = await scan_skills(temp_skills_dir)
        assert skills[0].path == skill_md
        assert skills[0].content == skill_md.read_text()


# -- search_similar (async, needs embeddings) ---------------------------------


class TestSearchSimilar:
    def _make_skills(self) -> list[SkillInfo]:
        return [
            SkillInfo(name="git-api", description="Interact with Git and GitHub APIs", path=Path("x"), content="", has_rules=False),
            SkillInfo(name="deploy-stuff", description="Deploy applications to Kubernetes with helm charts", path=Path("x"), content="", has_rules=False),
            SkillInfo(name="debug-tool", description="Debug and troubleshoot build failures in CI/CD pipelines", path=Path("x"), content="", has_rules=False),
        ]

    async def test_similar_query_ranks_highest(self):
        skills = self._make_skills()
        results = await search_similar("git operations and github interaction", skills)
        assert results[0][0].name == "git-api"

    async def test_empty_skills_returns_empty(self):
        results = await search_similar("anything", [])
        assert results == []

    async def test_top_k_respected(self):
        skills = self._make_skills()
        results = await search_similar("deploy", skills, top_k=2)
        assert len(results) == 2

    async def test_returns_scores_between_0_and_1(self):
        skills = self._make_skills()
        results = await search_similar("debug CI failures", skills)
        for _, score in results:
            assert 0 <= score <= 1

    async def test_exact_match_high_score(self):
        skills = [
            SkillInfo(name="test", description="Interact with Git and GitHub APIs", path=Path("x"), content="", has_rules=False),
            SkillInfo(name="other", description="Completely unrelated topic about cooking recipes", path=Path("x"), content="", has_rules=False),
        ]
        results = await search_similar("Interact with Git and GitHub APIs", skills)
        assert results[0][0].name == "test"
        assert results[0][1] > 0.9


# -- Session tracking (async, needs DBs) ---------------------------------------


class TestGetUnprocessedSessions:
    async def test_empty_processed_all_returned(self, skills_db_path, db_path):
        sessions = await get_unprocessed_sessions("synthesize", limit=3, skills_db_path=skills_db_path, opencode_db_path=db_path)
        assert len(sessions) == 3

    async def test_processed_excluded(self, skills_db_path, db_path):
        await mark_sessions_processed("synthesize", ["s1"], db_path=skills_db_path)
        sessions = await get_unprocessed_sessions("synthesize", limit=10, skills_db_path=skills_db_path, opencode_db_path=db_path)
        assert "s1" not in sessions

    async def test_synthesize_oldest_first(self, skills_db_path, db_path):
        sessions = await get_unprocessed_sessions("synthesize", limit=2, skills_db_path=skills_db_path, opencode_db_path=db_path)
        assert sessions[0] == "s1"  # oldest time_created
        assert sessions[1] == "s2"

    async def test_evolve_newest_first(self, skills_db_path, db_path):
        sessions = await get_unprocessed_sessions("evolve", limit=2, skills_db_path=skills_db_path, opencode_db_path=db_path)
        assert sessions[0] == "s5"  # newest time_created
        assert sessions[1] == "s4"

    async def test_limit_respected(self, skills_db_path, db_path):
        sessions = await get_unprocessed_sessions("evolve", limit=1, skills_db_path=skills_db_path, opencode_db_path=db_path)
        assert len(sessions) == 1


class TestMarkSessionsProcessed:
    async def test_marks_and_detected(self, skills_db_path):
        await mark_sessions_processed("synthesize", ["s1"], db_path=skills_db_path)
        assert await is_session_processed("synthesize", "s1", db_path=skills_db_path) is True

    async def test_not_processed_by_default(self, skills_db_path):
        assert await is_session_processed("evolve", "nonexistent", db_path=skills_db_path) is False

    async def test_different_queues_independent(self, skills_db_path):
        await mark_sessions_processed("synthesize", ["s1"], db_path=skills_db_path)
        assert await is_session_processed("evolve", "s1", db_path=skills_db_path) is False

    async def test_duplicate_insert_idempotent(self, skills_db_path):
        await mark_sessions_processed("evolve", ["s1"], db_path=skills_db_path)
        await mark_sessions_processed("evolve", ["s1"], db_path=skills_db_path)
        assert await is_session_processed("evolve", "s1", db_path=skills_db_path) is True

    async def test_synthesize_stores_skill_name_and_action(self, skills_db_path):
        await mark_sessions_processed("synthesize", ["s1"], db_path=skills_db_path, skill_name="test-skill", action="created")
        conn = sqlite3.connect(skills_db_path)
        row = conn.execute("SELECT skill_name, action FROM processed_synthesize WHERE session_id = 's1'").fetchone()
        conn.close()
        assert row[0] == "test-skill"
        assert row[1] == "created"
