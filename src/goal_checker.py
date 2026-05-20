"""Check whether a conversation's messages indicate a goal was achieved."""

from dataclasses import dataclass

from src.llm import DEFAULT_MODEL, complete_tool

SYSTEM_PROMPT = """\
You are a goal-completion evaluator. You will receive a conversation (list of messages) \
and a goal description. Your job is to determine whether the conversation indicates \
that the goal was achieved.

Evaluate based on the evidence in the messages only. Do not speculate beyond what is shown."""

GOAL_RESULT_TOOL = {
    "name": "report_goal_result",
    "description": "Report whether the goal was achieved based on conversation evidence.",
    "input_schema": {
        "type": "object",
        "properties": {
            "achieved": {"type": "boolean", "description": "Whether the goal was achieved"},
            "reasoning": {"type": "string", "description": "One or two sentences explaining why"},
        },
        "required": ["achieved", "reasoning"],
    },
}


@dataclass
class GoalResult:
    achieved: bool
    reasoning: str


def _format_messages_for_prompt(messages: list[dict]) -> str:
    """Render messages into a readable transcript for the evaluator."""
    lines: list[str] = []
    for i, msg in enumerate(messages, 1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            # handle structured content blocks
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block["text"])
                elif isinstance(block, str):
                    parts.append(block)
            content = "\n".join(parts)
        lines.append(f"--- Message {i} ({role}) ---\n{content}")
    return "\n\n".join(lines)


async def check_goal_achieved(
    messages: list[dict],
    goal: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
) -> GoalResult:
    """Determine whether the messages indicate the goal was achieved.

    Args:
        messages: List of message dicts with "role" and "content" keys.
        goal: Plain text description of what should have been accomplished.
        model: Model identifier to use.
        max_tokens: Maximum tokens in the response.

    Returns:
        GoalResult with achieved bool and reasoning string.
    """
    transcript = _format_messages_for_prompt(messages)

    result = await complete_tool(
        messages=[
            {
                "role": "user",
                "content": f"## Goal\n{goal}\n\n## Conversation\n{transcript}",
            }
        ],
        tool=GOAL_RESULT_TOOL,
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
    )

    return GoalResult(
        achieved=result["achieved"],
        reasoning=result["reasoning"],
    )
