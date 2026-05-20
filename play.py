"""List OpenCode conversations and extract goals from sessions."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from src.goal_checker import check_goal_achieved
from src.goal_clusterer import cluster_goals
from src.goal_extractor import extract_goals
from src.opencode_db import get_messages_for_session, get_sessions, slice_messages


def fmt_time(ts: str | int | float) -> str:
    if not ts:
        return "n/a"
    if isinstance(ts, (int, float)):
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    else:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return dt.astimezone().strftime("%Y-%m-%d %H:%M")


def fmt_cost(cost: float) -> str:
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


async def process_goals(session_id: str, check: bool = False) -> None:
    """Extract goals from a session and optionally check if achieved."""
    print(f"Extracting goals from session {session_id}...")
    goals = await extract_goals(session_id)

    if not check:
        for i, g in enumerate(goals, 1):
            print(f"  {i}. [{g.message_range}] {g.title}")
            print(f"     {g.description}")
        return

    all_messages = await get_messages_for_session(session_id)
    print(f"Loaded {len(all_messages)} messages. Checking {len(goals)} goals...\n")

    for i, g in enumerate(goals, 1):
        msgs_slice = slice_messages(all_messages, g.message_range)
        goal_text = f"{g.title}: {g.description}"
        result = await check_goal_achieved(msgs_slice, goal_text)
        status = "ACHIEVED" if result.achieved else "NOT ACHIEVED"
        print(f"  {i}. [{g.message_range}] {g.title}")
        print(f"     {g.description}")
        print(f"     -> {status}: {result.reasoning}")


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="List OpenCode conversations")
    parser.add_argument("-n", "--limit", type=int, default=30, help="Number of sessions to show")
    parser.add_argument("--dir", type=str, default=None, help="Filter by directory (substring match)")
    parser.add_argument("--agent", type=str, default=None, help="Filter by agent name")
    parser.add_argument("--goals", type=str, default=None, metavar="SESSION_ID_OR_INDEX",
                        help="Extract goals from a session (pass session ID or row index from listing)")
    parser.add_argument("--goals-file", type=str, default=None, metavar="PATH",
                        help="Extract goals from multiple sessions (file with one session ID per line)")
    parser.add_argument("--check", action="store_true",
                        help="With --goals/--goals-file: also check if each goal was achieved")
    parser.add_argument("--cluster", action="store_true",
                        help="With --goals-file: cluster extracted goals by similarity")
    parser.add_argument("--min-cluster-size", type=int, default=5,
                        help="Minimum cluster size; smaller clusters merge (default: 5)")
    parser.add_argument("--max-cluster-size", type=int, default=100,
                        help="Maximum cluster size; larger clusters split (default: 100)")
    args = parser.parse_args()

    # Handle --goals-file mode
    if args.goals_file is not None:
        lines = Path(args.goals_file).read_text().splitlines()
        session_ids = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
        print(f"Processing {len(session_ids)} sessions from {args.goals_file}\n")

        if args.cluster:
            all_goals: list[str] = []
            for sid in session_ids:
                goals = await extract_goals(sid)
                all_goals.extend(f"{g.title}: {g.description}" for g in goals)
            print(f"Collected {len(all_goals)} goals. Clustering...\n")

            result = cluster_goals(
                all_goals,
                min_cluster_size=args.min_cluster_size,
                max_cluster_size=args.max_cluster_size,
            )
            for cid, goals in sorted(result.clusters.items()):
                print(f"Cluster {cid + 1} ({len(goals)} goals):")
                for j, g in enumerate(goals, 1):
                    print(f"  {j}. {g}")
                print()
            print(f"{len(result.clusters)} clusters, {len(all_goals)} goals total")
        else:
            for sid in session_ids:
                print(f"=== Session {sid} ===")
                await process_goals(sid, check=args.check)
                print()
        return

    # Handle --goals mode
    if args.goals is not None:
        sid = args.goals
        # If it looks like a small number, treat as index into filtered list
        if sid.isdigit() and int(sid) <= 500:
            idx = int(sid) - 1
            has_filter = args.dir or args.agent
            all_sessions = await get_sessions(limit=2000 if has_filter else 500)
            if args.dir:
                all_sessions = [s for s in all_sessions if args.dir in s.directory]
            if args.agent:
                all_sessions = [s for s in all_sessions if s.agent == args.agent]
            if idx >= len(all_sessions):
                print(f"Index {sid} out of range (only {len(all_sessions)} sessions)")
                return
            sid = all_sessions[idx].id
            print(f"Session: {all_sessions[idx].title} ({sid})")

        await process_goals(sid, check=args.check)
        return

    has_filter = args.dir or args.agent
    sessions = await get_sessions(limit=args.limit if not has_filter else 2000)

    if args.dir:
        sessions = [s for s in sessions if args.dir in s.directory]
    if args.agent:
        sessions = [s for s in sessions if s.agent == args.agent]
    sessions = sessions[: args.limit]

    print(f"{'#':<4} {'Updated':<17} {'Agent':<14} {'Model':<30} {'Msgs':>5} {'Cost':>9} {'In':>7} {'Out':>7}  Title")
    print("-" * 130)

    for i, s in enumerate(sessions, 1):
        title = s.title[:60] if len(s.title) > 60 else s.title
        print(
            f"{i:<4} {fmt_time(s.time_updated):<17} {s.agent:<14} {s.model_id:<30} {s.message_count:>5} {fmt_cost(s.cost):>9} {fmt_tokens(s.tokens_input):>7} {fmt_tokens(s.tokens_output):>7}  {title}"
        )

    print(f"\nShowing {len(sessions)} sessions (total in DB: query limited)")


if __name__ == "__main__":
    asyncio.run(main())
