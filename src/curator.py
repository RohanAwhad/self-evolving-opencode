"""Curator — per-skill ADD-only rule synthesis from reflector insights.

Runs per skill within a goal cluster, after all threads have been reflected.
Synthesizes new rules, deduplicates against existing rules, and outputs
ADD_RULE operations. Never modifies or deletes existing rules.
"""

import json
import textwrap
from dataclasses import dataclass
from typing import Literal

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.llm import DEFAULT_MODEL, complete
from src.skill_registry import SkillInfo
from src.skill_rules import RuleStats

EMBEDDING_MODEL = "all-mpnet-base-v2"


@dataclass
class CuratorOperation:
    type: Literal["ADD_RULE"]
    target_skill: str
    content: str
    reasoning: str


CURATOR_SYSTEM = textwrap.dedent("""\
    You are a skill curator. You will receive:
    1. A skill (its current rules and workflow)
    2. A list of new insights from recent conversation threads
    3. Statistics on existing rules (helpful/harmful counts)

    Your job:
    - Synthesize new insights into concrete, actionable rules
    - Do NOT modify or delete existing rules (append-only)
    - If an insight is already covered by an existing rule, skip it
    - If multiple insights say the same thing, combine them into one rule
    - Each rule should be a clear, executable instruction

    Output ONLY a JSON array of ADD_RULE operations:
    [
      {"type": "ADD_RULE", "content": "Rule text", "reasoning": "Why this rule helps"}
    ]
    """)


async def curate_skill(
    skill_name: str,
    insights: list[str],
    current_skill: SkillInfo,
    rule_stats: RuleStats,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
) -> list[CuratorOperation]:
    insights_text = "\n".join(f"- {insight}" for insight in insights)
    stats_text = (
        f"Rule stats: {rule_stats.total} total, "
        f"{rule_stats.high_performing} high-performing, "
        f"{rule_stats.suspicious} suspicious, "
        f"{rule_stats.unused} unused, "
        f"avg helpful={rule_stats.average_helpful:.1f}, "
        f"avg harmful={rule_stats.average_harmful:.1f}"
    )

    prompt = (
        f"Skill: {skill_name}\n\n"
        f"Current skill content:\n```markdown\n{current_skill.content}\n```\n\n"
        f"New insights:\n{insights_text}\n\n"
        f"{stats_text}\n\n"
        f"Synthesize new rules from these insights."
    )

    response = await complete(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=max_tokens,
        system=CURATOR_SYSTEM,
    )

    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    data = json.loads(cleaned)

    ops: list[CuratorOperation] = []
    existing_rules: list[str] = _extract_rule_contents(current_skill.content)

    for item in data:
        content = item.get("content", "")
        if not content.strip():
            continue
        # dedup against existing rules
        if _is_duplicate(content, existing_rules):
            continue
        ops.append(
            CuratorOperation(
                type="ADD_RULE",
                target_skill=skill_name,
                content=content,
                reasoning=item.get("reasoning", ""),
            )
        )
        existing_rules.append(content)

    return ops


# ---------------------------------------------------------------------------
# Rule extraction and dedup
# ---------------------------------------------------------------------------


def _extract_rule_contents(skill_md: str) -> list[str]:
    """Extract rule content strings from the ## Rules section of a SKILL.md."""
    rules: list[str] = []
    in_rules = False
    for line in skill_md.split("\n"):
        if line.startswith("## Rules"):
            in_rules = True
            continue
        if in_rules:
            if line.startswith("## "):
                break
            stripped = line.strip()
            if stripped.startswith("- [") and "]" in stripped:
                content = stripped.split("] ", 1)[1] if "] " in stripped else stripped
                rules.append(content)
    return rules


_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def _is_duplicate(new_content: str, existing_contents: list[str], threshold: float = 0.90) -> bool:
    """Check if new rule content is too similar to an existing rule."""
    if not existing_contents:
        return False
    model = _get_model()
    new_emb = model.encode([new_content], show_progress_bar=False, convert_to_numpy=True)
    existing_embs = model.encode(existing_contents, show_progress_bar=False, convert_to_numpy=True)
    sims = cosine_similarity(new_emb, existing_embs)[0]
    return bool((sims > threshold).any())
