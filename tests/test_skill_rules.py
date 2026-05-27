"""Tests for src/skill_rules.py — rule DB CRUD operations."""

import sqlite3
from pathlib import Path

from src.skill_rules import (
    RuleRow,
    RuleStats,
    RuleTag,
    get_max_rule_id,
    get_rule_stats,
    get_rules_for_skill,
    insert_rules,
    next_rule_ids,
    update_counters,
)


# -- insert_rules / get_rules_for_skill ----------------------------------------


class TestInsertAndGet:
    async def test_insert_and_retrieve(self, skills_db_path):
        await insert_rules(
            "test-skill",
            [("test-skill-00001", "Always verify"), ("test-skill-00002", "Never assume")],
            db_path=skills_db_path,
        )
        rules = await get_rules_for_skill("test-skill", db_path=skills_db_path)
        assert len(rules) == 2
        assert rules[0].id == "test-skill-00001"
        assert rules[0].content == "Always verify"
        assert rules[0].skill_name == "test-skill"
        assert rules[0].helpful_count == 0
        assert rules[0].harmful_count == 0

    async def test_empty_skill_returns_empty(self, skills_db_path):
        rules = await get_rules_for_skill("nonexistent", db_path=skills_db_path)
        assert rules == []

    async def test_multiple_skills_isolated(self, skills_db_path):
        await insert_rules("skill-a", [("skill-a-00001", "Rule A")], db_path=skills_db_path)
        await insert_rules("skill-b", [("skill-b-00001", "Rule B")], db_path=skills_db_path)
        assert len(await get_rules_for_skill("skill-a", db_path=skills_db_path)) == 1
        assert len(await get_rules_for_skill("skill-b", db_path=skills_db_path)) == 1


# -- update_counters -----------------------------------------------------------


class TestUpdateCounters:
    async def test_followed_helpful_increments(self, skills_db_path):
        await insert_rules(
            "s", [("s-00001", "Test rule")], db_path=skills_db_path
        )
        await update_counters(
            [RuleTag(rule_id="s-00001", tag="followed_helpful", session_id="ses1")],
            db_path=skills_db_path,
        )
        rules = await get_rules_for_skill("s", db_path=skills_db_path)
        assert rules[0].helpful_count == 1
        assert rules[0].harmful_count == 0

    async def test_followed_harmful_increments(self, skills_db_path):
        await insert_rules("s", [("s-00001", "Bad rule")], db_path=skills_db_path)
        await update_counters(
            [RuleTag(rule_id="s-00001", tag="followed_harmful", session_id="ses1")],
            db_path=skills_db_path,
        )
        rules = await get_rules_for_skill("s", db_path=skills_db_path)
        assert rules[0].harmful_count == 1
        assert rules[0].helpful_count == 0

    async def test_not_followed_increments_harmful(self, skills_db_path):
        await insert_rules("s", [("s-00001", "Rule")], db_path=skills_db_path)
        await update_counters(
            [RuleTag(rule_id="s-00001", tag="not_followed", session_id="ses1")],
            db_path=skills_db_path,
        )
        rules = await get_rules_for_skill("s", db_path=skills_db_path)
        assert rules[0].harmful_count == 1

    async def test_irrelevant_no_change(self, skills_db_path):
        await insert_rules("s", [("s-00001", "Rule")], db_path=skills_db_path)
        await update_counters(
            [RuleTag(rule_id="s-00001", tag="irrelevant", session_id="ses1")],
            db_path=skills_db_path,
        )
        rules = await get_rules_for_skill("s", db_path=skills_db_path)
        assert rules[0].helpful_count == 0
        assert rules[0].harmful_count == 0

    async def test_batch_update(self, skills_db_path):
        await insert_rules(
            "s",
            [("s-00001", "R1"), ("s-00002", "R2")],
            db_path=skills_db_path,
        )
        await update_counters(
            [
                RuleTag(rule_id="s-00001", tag="followed_helpful", session_id="ses1"),
                RuleTag(rule_id="s-00002", tag="followed_harmful", session_id="ses1"),
                RuleTag(rule_id="s-00001", tag="followed_helpful", session_id="ses2"),
            ],
            db_path=skills_db_path,
        )
        rules = {r.id: r for r in await get_rules_for_skill("s", db_path=skills_db_path)}
        assert rules["s-00001"].helpful_count == 2
        assert rules["s-00002"].harmful_count == 1

    async def test_nonexistent_rule_id_silent(self, skills_db_path):
        await update_counters(
            [RuleTag(rule_id="nonexistent", tag="followed_helpful", session_id="ses1")],
            db_path=skills_db_path,
        )


# -- get_max_rule_id -----------------------------------------------------------


class TestGetMaxRuleId:
    async def test_empty_returns_zero(self, skills_db_path):
        assert await get_max_rule_id("nonexistent", db_path=skills_db_path) == 0

    async def test_returns_highest(self, skills_db_path):
        await insert_rules(
            "s", [("s-00001", "R1"), ("s-00005", "R5")], db_path=skills_db_path
        )
        assert await get_max_rule_id("s", db_path=skills_db_path) == 5

    async def test_isolated_by_skill(self, skills_db_path):
        await insert_rules("a", [("a-00010", "R")], db_path=skills_db_path)
        await insert_rules("b", [("b-00003", "R")], db_path=skills_db_path)
        assert await get_max_rule_id("a", db_path=skills_db_path) == 10
        assert await get_max_rule_id("b", db_path=skills_db_path) == 3


# -- get_rule_stats ------------------------------------------------------------


class TestGetRuleStats:
    async def test_empty_returns_zeros(self, skills_db_path):
        stats = await get_rule_stats("nonexistent", db_path=skills_db_path)
        assert stats == RuleStats(
            total=0, high_performing=0, suspicious=0, unused=0,
            average_helpful=0.0, average_harmful=0.0,
        )

    async def test_high_performing_detected(self, skills_db_path):
        # helpful > harmful * 2 → high_performing
        # Set 3 helpful, 1 harmful: 3 > 1*2 = 2 ✓
        conn = sqlite3.connect(skills_db_path)
        conn.executemany(
            "INSERT INTO rules (id, skill_name, content, helpful_count, harmful_count, created_at, updated_at) VALUES (?, ?, ?, ?, ?, '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z')",
            [
                ("s-00001", "s", "R1", 3, 1),
                ("s-00002", "s", "R2", 0, 0),
            ],
        )
        conn.commit()
        conn.close()
        stats = await get_rule_stats("s", db_path=skills_db_path)
        assert stats.total == 2
        assert stats.high_performing == 1
        assert stats.unused == 1

    async def test_suspicious_detected(self, skills_db_path):
        # harmful > helpful * 3 → suspicious
        conn = sqlite3.connect(skills_db_path)
        conn.executemany(
            "INSERT INTO rules (id, skill_name, content, helpful_count, harmful_count, created_at, updated_at) VALUES (?, ?, ?, ?, ?, '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z')",
            [
                ("s-00001", "s", "R1", 1, 4),  # 4 > 1*3 = 3 → suspicious
            ],
        )
        conn.commit()
        conn.close()
        stats = await get_rule_stats("s", db_path=skills_db_path)
        assert stats.suspicious == 1

    async def test_averages_computed(self, skills_db_path):
        conn = sqlite3.connect(skills_db_path)
        conn.executemany(
            "INSERT INTO rules (id, skill_name, content, helpful_count, harmful_count, created_at, updated_at) VALUES (?, ?, ?, ?, ?, '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z')",
            [
                ("s-00001", "s", "R1", 4, 2),
                ("s-00002", "s", "R2", 2, 0),
            ],
        )
        conn.commit()
        conn.close()
        stats = await get_rule_stats("s", db_path=skills_db_path)
        assert stats.average_helpful == 3.0
        assert stats.average_harmful == 1.0


# -- next_rule_ids -------------------------------------------------------------


class TestNextRuleIds:
    async def test_returns_correct_number(self, skills_db_path):
        ids = await next_rule_ids("skill", 3, db_path=skills_db_path)
        assert ids == ["skill-00001", "skill-00002", "skill-00003"]

    async def test_accounts_for_existing(self, skills_db_path):
        await insert_rules(
            "skill", [("skill-00001", "R1")], db_path=skills_db_path
        )
        ids = await next_rule_ids("skill", 2, db_path=skills_db_path)
        assert ids == ["skill-00002", "skill-00003"]
