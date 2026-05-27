"""Summarize a conversation from rich messages into a markdown report."""

from loguru import logger

from src.llm import DEFAULT_MODEL, complete_tool


SUMMARIZE_TOOL = {
    "name": "summarize_conversation",
    "description": "Return a structured summary of the conversation.",
    "input_schema": {
        "type": "object",
        "properties": {
            "goal": {"type": "string", "description": "What the user was trying to achieve (1-2 sentences)"},
            "intent": {"type": "string", "description": "Underlying motivation — why they wanted this"},
            "what_happened": {"type": "string", "description": "Narrative of the conversation flow: attempts, pivots, decisions"},
            "user_messages": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Numbered summaries of each user message",
            },
            "assistant_actions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Numbered summaries of each assistant action",
            },
            "tool_usage": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key tool calls: tool name, input summary, result summary",
            },
            "outcome": {"type": "string", "description": "Final result — completed? partially? what was produced?"},
            "evaluation_criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Criteria to judge if this conversation was successful",
            },
        },
        "required": ["goal", "intent", "what_happened", "user_messages", "assistant_actions", "tool_usage", "outcome", "evaluation_criteria"],
    },
}

SYSTEM_PROMPT = """\
You are a conversation analyst. You receive a transcript of a coding-assistant \
conversation (with tool calls and their results) and produce a detailed summary.

Analyze the conversation and extract each section carefully. \
For tool_usage, skip trivial/duplicate calls — focus on ones that drove the conversation forward."""


def _format_rich_messages(messages: list[dict]) -> str:
    """Format rich messages (with parts) into a readable transcript string."""
    lines: list[str] = []
    for i, msg in enumerate(messages, 1):
        role = msg.get("role", "unknown")
        lines.append(f"--- Message {i} ({role}) ---")

        for part in msg.get("parts", []):
            ptype = part.get("type", "")

            if ptype == "text":
                lines.append(part["text"])

            elif ptype == "tool":
                tool_name = part.get("tool", "unknown")
                status = part.get("status", "")
                title = part.get("title", "")
                tool_input = part.get("input", {})
                output = part.get("output", "")

                header = f"[tool: {tool_name}]"
                if title:
                    header += f" — {title}"
                if status:
                    header += f" ({status})"
                lines.append(header)

                # compact input representation
                if tool_input:
                    input_parts: list[str] = []
                    for k, v in tool_input.items():
                        v_str = str(v)
                        if len(v_str) > 200:
                            v_str = v_str[:200] + "..."
                        input_parts.append(f"  {k}: {v_str}")
                    lines.append("  Input:")
                    lines.extend(input_parts)

                # truncated output
                if output:
                    out_preview = output if len(output) <= 500 else output[:500] + "..."
                    lines.append(f"  Output: {out_preview}")

            elif ptype == "reasoning":
                lines.append(f"[reasoning] {part.get('text', '')}")

        lines.append("")  # blank line between messages

    return "\n".join(lines)


def _build_markdown(data: dict) -> str:
    """Assemble structured tool output into markdown."""
    sections: list[str] = []
    sections.append(f"## Goal\n{data.get('goal', '')}")
    sections.append(f"## Intent\n{data.get('intent', '')}")
    sections.append(f"## What Happened\n{data.get('what_happened', '')}")

    user_msgs = data.get("user_messages", [])
    if user_msgs:
        items = "\n".join(f"{i}. {m}" for i, m in enumerate(user_msgs, 1))
        sections.append(f"## User Messages\n{items}")

    actions = data.get("assistant_actions", [])
    if actions:
        items = "\n".join(f"{i}. {a}" for i, a in enumerate(actions, 1))
        sections.append(f"## Assistant Actions\n{items}")

    tools = data.get("tool_usage", [])
    if tools:
        items = "\n".join(f"{i}. {t}" for i, t in enumerate(tools, 1))
        sections.append(f"## Tool Usage\n{items}")

    sections.append(f"## Outcome\n{data.get('outcome', '')}")

    criteria = data.get("evaluation_criteria", [])
    if criteria:
        items = "\n".join(f"- {c}" for c in criteria)
        sections.append(f"## Evaluation Criteria\n{items}")

    return "\n\n".join(sections)


async def summarize_conversation(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 8192,
    system: str | None = None,
) -> str:
    """Summarize a conversation from rich messages into markdown.

    Args:
        messages: Rich messages from get_rich_messages_for_session().
                  Each dict has {"role": str, "parts": [{"type": "text"|"tool"|"reasoning", ...}]}.
        model: LLM model to use.
        max_tokens: Max output tokens.
        system: Override system prompt (uses default if None).

    Returns:
        Markdown string with conversation summary.
    """
    transcript = _format_rich_messages(messages)
    logger.debug("Transcript length: {} chars, {} messages", len(transcript), len(messages))

    result = await complete_tool(
        messages=[{"role": "user", "content": f"Summarize this conversation:\n\n{transcript}"}],
        tool=SUMMARIZE_TOOL,
        model=model,
        max_tokens=max_tokens,
        system=system or SYSTEM_PROMPT,
    )

    return _build_markdown(result)
