"""LLM-based skill synthesizer — extracts workflows from conversation clusters.

Takes a cluster of goals + thread summaries, synthesizes a complete SKILL.md
(frontmatter + workflow + rules). Handles both new skill creation and existing
skill workflow updates.
"""

import textwrap

from src.goal_extractor import Goal
from src.llm import DEFAULT_MODEL, complete
from src.skill_registry import SkillInfo

SYNTHESIZE_NEW_SYSTEM = textwrap.dedent("""\
    You are a skill synthesizer. You will receive:
    1. A cluster of related user goals
    2. Summaries of conversation threads that achieved these goals

    Your job:
    - Identify the common workflow pattern across all threads
    - Extract what the user consistently does, in what order
    - Identify decision points, checkpoints, and repeatable patterns
    - Write a concise but complete SKILL.md

    The output MUST be a complete SKILL.md with YAML frontmatter followed by
    markdown sections. Format:

    ---
    name: kebab-case-name
    description: Detailed description (1-3 sentences, imperative phrasing)
    ---

    ## Workflow
    ### Phase 1: ...
    1. ...
    2. ...

    ### Phase 2: ...
    ...

    ## Rules
    - [skillname-00001] Rule text
    - [skillname-00002] Rule text

    Rules are concrete patterns the agent should follow. Include rule IDs in
    brackets. Do NOT include counters in the markdown.
    """)


SYNTHESIZE_UPDATE_SYSTEM = textwrap.dedent("""\
    You are a skill updater. You will receive:
    1. An existing SKILL.md
    2. A cluster of related user goals
    3. Summaries of conversation threads that achieved these goals

    Your job:
    - Refine the ## Workflow section with new insights from the threads
    - Preserve the existing ## Rules section exactly as-is
    - Keep the existing frontmatter (you may update the description)
    - Do NOT modify or delete any rules

    Output the full updated SKILL.md.
    """)


async def synthesize_skill(
    cluster_id: int,
    goals: list[Goal],
    thread_summaries: list[str],
    existing_skill: SkillInfo | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 8192,
) -> str:
    goals_text = "\n".join(f"- {g.title}: {g.description}" for g in goals)
    threads_text = "\n\n---\n\n".join(
        f"## Thread {i + 1}\n{s}" for i, s in enumerate(thread_summaries)
    )

    if existing_skill is not None:
        system = SYNTHESIZE_UPDATE_SYSTEM
        prompt = (
            f"Existing skill:\n```markdown\n{existing_skill.content}\n```\n\n"
            f"New threads:\n{threads_text}\n\n"
            f"Update the ## Workflow section. Preserve ## Rules exactly."
        )
    else:
        system = SYNTHESIZE_NEW_SYSTEM
        prompt = (
            f"Goals (cluster {cluster_id}):\n{goals_text}\n\n"
            f"Thread summaries:\n{threads_text}\n\n"
            f"Synthesize a complete SKILL.md."
        )

    response = await complete(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=max_tokens,
        system=system,
    )

    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return cleaned
