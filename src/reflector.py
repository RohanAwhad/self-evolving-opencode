"""Reflector — per-thread rule tagging and insight extraction.

Evaluates a conversation thread against skill rules, tags each rule, and
extracts new insights grouped by skill.
"""

import json
import textwrap
from dataclasses import dataclass
from typing import Literal

from src.llm import DEFAULT_MODEL, complete
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

    Output ONLY a JSON object:
    {
      "rule_tags": [{"rule_id": "skill-00001", "tag": "followed_helpful"}, ...],
      "insights_by_skill": {"skill-a": ["insight 1"], "skill-b": []}
    }
    """)


REFLECT_INSIGHT_SYSTEM = textwrap.dedent("""\
    You are a skill insight extractor. You will receive:
    1. A conversation summary
    2. A list of skill names that were active during this conversation

    Extract new insights — patterns, friction points, what worked, what didn't.
    Group insights by the skill they apply to.

    Output ONLY a JSON object:
    {
      "insights_by_skill": {"skill-a": ["insight 1"], "skill-b": []}
    }
    """)


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

    response = await complete(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=max_tokens,
        system=REFLECT_TAG_SYSTEM,
    )

    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    data = json.loads(cleaned)

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

    response = await complete(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=max_tokens,
        system=REFLECT_INSIGHT_SYSTEM,
    )

    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    data = json.loads(cleaned)

    insights = data.get("insights_by_skill", {})
    return Reflection(session_id=session_id, rule_tags=[], insights_by_skill=insights)
