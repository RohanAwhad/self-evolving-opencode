"""Summarize a conversation from rich messages into a markdown report."""

import re

from loguru import logger

from src.llm import DEFAULT_MODEL, complete


SUMMARY_START_TAG = "<|CONVERSATION_SUMMARY_SEA_START|>"
SUMMARY_END_TAG = "<|CONVERSATION_SUMMARY_SEA_END|>"

SYSTEM_PROMPT = """\
You are a conversation analyst. You receive a transcript of a coding-assistant \
conversation (with tool calls and their results) and produce a detailed markdown summary.

Your summary MUST be enclosed in these exact tags:
{start_tag}
<your markdown here>
{end_tag}

The markdown MUST contain these sections in order:

## Goal
What the user was ultimately trying to achieve. One or two sentences.

## Intent
The underlying motivation — why they wanted this, what problem they were solving.

## What Happened
A narrative of the conversation flow: what was attempted, what worked, what didn't, \
key decision points, and pivots.

## User Messages
A numbered list summarizing each user message (what they asked/said/clarified).

## Assistant Actions
A numbered list summarizing each assistant reply (what it did, proposed, or explained).

## Tool Usage
A numbered list of tool calls made, grouped logically. For each: the tool name, \
what it was called with (brief), and what the result was (brief). \
Skip trivial/duplicate calls — focus on ones that drove the conversation forward.

## Outcome
What was the final result — was the task completed? Partially? What was produced?

## Evaluation Criteria
A bullet list of criteria an evaluator could use to judge whether this conversation \
was successful. Derive these from the user's stated and implied requirements.
""".format(start_tag=SUMMARY_START_TAG, end_tag=SUMMARY_END_TAG)


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


def _extract_summary(response: str) -> str:
    """Extract markdown between the summary tags, or return raw response."""
    pattern = re.escape(SUMMARY_START_TAG) + r"(.*?)" + re.escape(SUMMARY_END_TAG)
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    logger.warning("Summary tags not found in LLM response, returning raw output")
    return response.strip()


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

    result = await complete(
        messages=[{"role": "user", "content": f"Summarize this conversation:\n\n{transcript}"}],
        model=model,
        max_tokens=max_tokens,
        system=system or SYSTEM_PROMPT,
    )

    return _extract_summary(result)
