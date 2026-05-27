"""Skill evolution pipeline — synthesize + evolve queues.

Orchestrates the full --evolve pipeline:
1. Synthesize queue (oldest-first): extract goals, cluster, synthesize skills
2. Evolve queue (newest-first): reflect on threads, curate rules

Sequential execution avoids race conditions on SKILL.md writes.
DRY_RUN=1 prevents all disk/SQLite writes. Output goes to stdout.
"""

import asyncio
import os
from collections import defaultdict
from pathlib import Path
from re import sub

from loguru import logger

from src.conversation_summarizer import summarize_conversation
from src.curator import curate_skill
from src.goal_clusterer import ClusterResult, cluster_goals
from src.goal_extractor import Goal, extract_goals
from src.opencode_db import (
    DB_PATH as OPENCODE_DB_PATH,
    get_rich_messages_for_session,
    get_skills_for_session,
    slice_messages,
)
from src.reflector import reflect_insight_only, reflect_on_thread
from src.skill_registry import (
    SKILLS_DIR_DEFAULT,
    SkillDecision,
    SkillInfo,
    decide_new_or_update,
    find_closest_skill,
    get_unprocessed_sessions,
    mark_sessions_processed,
    scan_skills,
)
from src.skill_rules import (
    SKILLS_DB_PATH,
    RuleRow,
    get_rule_stats,
    get_rules_for_skill,
    insert_rules,
    next_rule_ids,
    update_counters,
)
from src.skill_synthesizer import synthesize_skill

DRY_RUN = os.environ.get("DRY_RUN") == "1"


def _ensure_skills_dir(skills_dir: Path) -> None:
    if not DRY_RUN:
        skills_dir.mkdir(parents=True, exist_ok=True)


def _write_skill_md(skill_name: str, content: str, skills_dir: Path) -> None:
    if DRY_RUN:
        logger.info("DRY_RUN: would write {}/SKILL.md ({:,} chars)", skill_name, len(content))
        print(f"\n--- {skill_name}/SKILL.md ---\n{content}\n--- end ---\n")
        return
    skill_path = skills_dir / skill_name / "SKILL.md"
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    skill_path.write_text(content)
    logger.info("Wrote {}/SKILL.md ({:,} chars)", skill_name, len(content))


def _derive_skill_name(goals: list[Goal]) -> str:
    name = goals[0].title if goals else "new-skill"
    name = name.lower().strip()
    name = sub(r"[^a-z0-9]+", "-", name)
    return name[:64].strip("-")


# ---------------------------------------------------------------------------
# Synthesizer queue
# ---------------------------------------------------------------------------


async def _run_synthesizer(
    limit: int,
    skills_dir: Path,
    skills_db_path: Path,
    opencode_db_path: Path,
    max_concurrency: int = 8,
) -> None:
    session_ids = await get_unprocessed_sessions(
        "synthesize", limit=limit,
        skills_db_path=skills_db_path, opencode_db_path=opencode_db_path,
    )
    if not session_ids:
        logger.info("Synthesizer: no unprocessed sessions")
        return

    logger.info("Synthesizer: {} unprocessed sessions (oldest first)", len(session_ids))

    # Extract goals from all sessions in parallel (bounded by semaphore)
    sem = asyncio.Semaphore(max_concurrency)

    async def _extract_one(sid: str) -> tuple[str, list[Goal]]:
        async with sem:
            goals = await extract_goals(sid, db_path=opencode_db_path)
        return sid, goals

    results = await asyncio.gather(*(_extract_one(sid) for sid in session_ids))

    all_goal_texts: list[str] = []
    goal_sources: list[tuple[str, Goal]] = []
    sessions_with_goals: set[str] = set()
    for sid, goals in results:
        for g in goals:
            text = f"{g.title}: {g.description}"
            all_goal_texts.append(text)
            goal_sources.append((sid, g))
            sessions_with_goals.add(sid)

    # Mark sessions with no goals as processed
    no_goal_sids = [sid for sid in session_ids if sid not in sessions_with_goals]
    if no_goal_sids and not DRY_RUN:
        await mark_sessions_processed("synthesize", no_goal_sids, db_path=skills_db_path)

    if not all_goal_texts:
        logger.info("Synthesizer: no goals extracted from any session")
        return

    # Cluster goals
    result: ClusterResult = cluster_goals(all_goal_texts, min_cluster_size=3, max_cluster_size=20)

    # Scan existing skills once
    existing_skills = await scan_skills(skills_dir)
    skills_by_name = {s.name: s for s in existing_skills}

    text_to_source = dict(zip(all_goal_texts, goal_sources))
    processed_cluster_sids: set[str] = set()
    skills_created: list[str] = []
    skills_updated: list[str] = []

    for cid, cluster_goals_list in sorted(result.clusters.items()):
        # Collect sessions for this cluster
        cluster_sids: list[str] = []
        cluster_goals_objs: list[Goal] = []
        seen_sids: set[str] = set()
        for g_text in cluster_goals_list:
            source = text_to_source.get(g_text)
            if source and source[0] not in seen_sids:
                sid, goal = source
                cluster_sids.append(sid)
                cluster_goals_objs.append(goal)
                seen_sids.add(sid)

        if not cluster_sids:
            continue

        # Summarize ≤10 threads from this cluster
        summaries: list[str] = []
        summarized_sids: list[str] = []

        # Prepare data for all candidates
        summarize_candidates: list[tuple[str, Goal]] = []
        for sid in cluster_sids:
            if len(summarize_candidates) >= 10:
                break
            sid_goal = next((g for ss, g in goal_sources if ss == sid), None)
            if not sid_goal:
                continue
            summarize_candidates.append((sid, sid_goal))

        if not summarize_candidates:
            continue

        async def _summarize_one(sid: str, sid_goal: Goal) -> tuple[str, str | None]:
            async with sem:
                rich = await get_rich_messages_for_session(sid, db_path=opencode_db_path)
                thread = slice_messages(rich, sid_goal.message_range)
                if thread:
                    summary = await summarize_conversation(thread)
                    return sid, summary
            return sid, None

        summary_results = await asyncio.gather(*(_summarize_one(sid, g) for sid, g in summarize_candidates))
        for sid, summary in summary_results:
            if summary:
                summaries.append(summary)
                summarized_sids.append(sid)

        if not summaries:
            continue

        # Semantic search → decide new/update
        cluster_desc = " ".join(g.description for g in cluster_goals_objs)[:500]
        skill_name = _derive_skill_name(cluster_goals_objs)

        top_matches = await find_closest_skill(cluster_desc, skills_dir=skills_dir, top_k=3)
        decision = SkillDecision(action="new", target_skill=None, reasoning="")
        if top_matches and top_matches[0][1] > 0.5:
            decision = await decide_new_or_update(skill_name, cluster_desc, top_matches)

        if decision.action == "update" and decision.target_skill and decision.target_skill in skills_by_name:
            matched = skills_by_name[decision.target_skill]
            skill_content = await synthesize_skill(
                cluster_id=cid, goals=cluster_goals_objs,
                thread_summaries=summaries, existing_skill=matched,
            )
            _write_skill_md(decision.target_skill, skill_content, skills_dir)
            skills_updated.append(decision.target_skill)
        else:
            skill_content = await synthesize_skill(
                cluster_id=cid, goals=cluster_goals_objs, thread_summaries=summaries,
            )
            _write_skill_md(skill_name, skill_content, skills_dir)
            skills_created.append(skill_name)

        processed_cluster_sids.update(summarized_sids)

    # Mark sessions as processed
    all_processed = list(processed_cluster_sids | set(no_goal_sids))
    if all_processed and not DRY_RUN:
        await mark_sessions_processed("synthesize", all_processed, db_path=skills_db_path)

    logger.info(
        "Synthesizer done: {} skills created, {} updated, {} sessions processed",
        len(skills_created), len(skills_updated), len(all_processed),
    )


# ---------------------------------------------------------------------------
# Evolve queue
# ---------------------------------------------------------------------------


async def _run_evolve(
    limit: int,
    skills_dir: Path,
    skills_db_path: Path,
    opencode_db_path: Path,
    max_concurrency: int = 8,
) -> None:
    session_ids = await get_unprocessed_sessions(
        "evolve", limit=limit,
        skills_db_path=skills_db_path, opencode_db_path=opencode_db_path,
    )
    if not session_ids:
        logger.info("Evolve: no unprocessed sessions")
        return

    logger.info("Evolve: {} unprocessed sessions (newest first)", len(session_ids))

    existing_skills = await scan_skills(skills_dir)
    skills_by_name = {s.name: s for s in existing_skills}

    total_tagged = 0
    total_added = 0
    all_insights: dict[str, list[str]] = defaultdict(list)

    for sid in session_ids:
        skill_names = await get_skills_for_session(sid, db_path=opencode_db_path)
        if not skill_names:
            if not DRY_RUN:
                await mark_sessions_processed("evolve", [sid], db_path=skills_db_path)
            continue

        rich = await get_rich_messages_for_session(sid, db_path=opencode_db_path)
        if not rich:
            if not DRY_RUN:
                await mark_sessions_processed("evolve", [sid], db_path=skills_db_path)
            continue
        summary = await summarize_conversation(rich)

        skills_with_rules: list[tuple[str, list[RuleRow]]] = []
        for sn in skill_names:
            rules = await get_rules_for_skill(sn, db_path=skills_db_path)
            skills_with_rules.append((sn, rules))

        any_rules = any(rules for _, rules in skills_with_rules)
        if any_rules:
            reflection = await reflect_on_thread(sid, summary, skills_with_rules)
        else:
            reflection = await reflect_insight_only(sid, summary, skill_names)

        if reflection.rule_tags:
            await update_counters(reflection.rule_tags, db_path=skills_db_path)
            total_tagged += len(reflection.rule_tags)

        for sname, insights in reflection.insights_by_skill.items():
            all_insights[sname].extend(insights)

        if not DRY_RUN:
            await mark_sessions_processed("evolve", [sid], db_path=skills_db_path)

    # Curator per skill with insights
    for sname, insights in all_insights.items():
        if not insights:
            continue
        skill_info = skills_by_name.get(sname)
        if not skill_info:
            continue
        stats = await get_rule_stats(sname, db_path=skills_db_path)
        ops = await curate_skill(sname, insights, skill_info, stats)
        if not ops:
            continue

        ids = await next_rule_ids(sname, len(ops), db_path=skills_db_path)
        await insert_rules(sname, [(ids[i], op.content) for i, op in enumerate(ops)], db_path=skills_db_path)

        new_lines = [f"- [{ids[i]}] {op.content}" for i, op in enumerate(ops)]
        new_rules_block = "\n" + "\n".join(new_lines) + "\n"
        if "## Rules" in skill_info.content:
            updated_content = skill_info.content + new_rules_block
        else:
            updated_content = skill_info.content + "\n## Rules\n" + new_rules_block
        _write_skill_md(sname, updated_content, skills_dir)

        total_added += len(ops)

    logger.info(
        "Evolve done: {} rules tagged, {} rules added, {} sessions processed",
        total_tagged, total_added, len(session_ids),
    )


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


async def run_evolve(
    limit: int = 50,
    skills_dir: Path = SKILLS_DIR_DEFAULT,
    skills_db_path: Path = SKILLS_DB_PATH,
    opencode_db_path: Path = OPENCODE_DB_PATH,
    max_concurrency: int = 8,
) -> None:
    _ensure_skills_dir(skills_dir)
    logger.info("=== Skill Evolution (DRY_RUN=%s) ===", DRY_RUN)

    await _run_synthesizer(limit, skills_dir, skills_db_path, opencode_db_path, max_concurrency)
    await _run_evolve(limit, skills_dir, skills_db_path, opencode_db_path, max_concurrency)

    logger.info("=== Skill Evolution complete ===")
