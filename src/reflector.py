"""Reflector — per-thread rule tagging and insight extraction.

Evaluates a conversation thread against skill rules, tags each rule, and
extracts new insights grouped by skill.
"""

import textwrap
from dataclasses import dataclass
from typing import Literal

from src.llm import DEFAULT_MODEL, complete_tool
from src.skill_rules import RuleRow, RuleTag


@dataclass
class Reflection:
    session_id: str
    rule_tags: list[RuleTag]
    insights_by_skill: dict[str, list[str]]


REFLECT_TAG_SYSTEM = textwrap.dedent("""\
    You are a skill rule evaluator. You will receive:
    1. A conversation summary
    2. A set of rules from skills that were active during this conversation

    For each rule, tag it as one of:
    - irrelevant: The rule doesn't apply to this conversation
    - followed_helpful: The agent followed this rule and it helped
    - followed_harmful: The agent followed this rule but it caused harm
    - not_followed: The rule was relevant but the agent didn't follow it

    Also identify new insights — patterns or lessons from this conversation
    that aren't covered by the existing rules. Group insights by skill.
    """)

REFLECT_TAG_TOOL = {
    "name": "reflect_on_rules",
    "description": "Tag each rule and extract new insights grouped by skill.",
    "input_schema": {
        "type": "object",
        "properties": {
            "rule_tags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "rule_id": {"type": "string", "description": "The rule ID (e.g. skill-00001)"},
                        "tag": {"type": "string", "enum": ["irrelevant", "followed_helpful", "followed_harmful", "not_followed"]},
                    },
                    "required": ["rule_id", "tag"],
                },
            },
            "insights_by_skill": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "description": "New insights grouped by skill name",
            },
        },
        "required": ["rule_tags", "insights_by_skill"],
    },
}

REFLECT_INSIGHT_SYSTEM = textwrap.dedent("""\
    You are a skill insight extractor. You will receive:
    1. A conversation summary
    2. A list of skill names that were active during this conversation

    Extract new insights — patterns, friction points, what worked, what didn't.
    Group insights by the skill they apply to.
    """)

REFLECT_INSIGHT_TOOL = {
    "name": "extract_insights",
    "description": "Extract new insights grouped by skill name.",
    "input_schema": {
        "type": "object",
        "properties": {
            "insights_by_skill": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "description": "New insights grouped by skill name",
            },
        },
        "required": ["insights_by_skill"],
    },
}


def _format_rules_for_prompt(skills: list[tuple[str, list[RuleRow]]]) -> str:
    lines: list[str] = []
    for skill_name, rules in skills:
        lines.append(f"[{skill_name}]:")
        if not rules:
            lines.append("  (no rules)")
        for r in rules:
            lines.append(f"  {r.id}: {r.content}")
    return "\n".join(lines)


async def reflect_on_thread(
    session_id: str,
    thread_summary: str,
    skills: list[tuple[str, list[RuleRow]]],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
) -> Reflection:
    rules_text = _format_rules_for_prompt(skills)

    prompt = (
        f"Session: {session_id}\n\n"
        f"Conversation summary:\n{thread_summary}\n\n"
        f"Rules:\n{rules_text}\n\n"
        f"Evaluate each rule and extract new insights."
    )

    data = await complete_tool(
        messages=[{"role": "user", "content": prompt}],
        tool=REFLECT_TAG_TOOL,
        model=model,
        max_tokens=max_tokens,
        system=REFLECT_TAG_SYSTEM,
    )

    tags = [
        RuleTag(rule_id=rt["rule_id"], tag=rt["tag"], session_id=session_id)
        for rt in data.get("rule_tags", [])
    ]
    insights = data.get("insights_by_skill", {})
    return Reflection(session_id=session_id, rule_tags=tags, insights_by_skill=insights)


async def reflect_insight_only(
    session_id: str,
    thread_summary: str,
    skill_names: list[str],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
) -> Reflection:
    skills_text = "\n".join(f"- {s}" for s in skill_names)

    prompt = (
        f"Session: {session_id}\n\n"
        f"Conversation summary:\n{thread_summary}\n\n"
        f"Active skills:\n{skills_text}\n\n"
        f"Extract new insights grouped by skill."
    )

    data = await complete_tool(
        messages=[{"role": "user", "content": prompt}],
        tool=REFLECT_INSIGHT_TOOL,
        model=model,
        max_tokens=max_tokens,
        system=REFLECT_INSIGHT_SYSTEM,
    )

    insights = data.get("insights_by_skill", {})
    return Reflection(session_id=session_id, rule_tags=[], insights_by_skill=insights)
