"""A minimal faithful Hermes host simulator for engine-level tests.

Mirrors the verified host integration contract:
- the conversation list carries full persistence fields (message_uid, timestamp, ...);
- the request copy is ``[system] + one rendered image per conversation message``
  with PERSISTENCE_ONLY_MESSAGE_FIELDS stripped and api_content popped;
- ``select_context`` receives both lists plus budget_tokens = engine.context_length.
"""

from __future__ import annotations

import json
from typing import Any

# Fields the host strips from every outgoing request copy.
PERSISTENCE_FIELDS = (
    "timestamp", "display_kind", "display_metadata", "_row_id", "_submit_row_session_id",
    "_merged_turn_prefix", "message_uid", "_absorbed_message_uids", "_tool_call_uids",
    "_tool_call_uid", "api_content", "_db_row_snapshot", "_canonical_row",
    "_restored_from_store",
)

SYSTEM_PROMPT = "You are a helpful assistant."


def make_turn(
    index: int,
    *,
    user_text: str | None = None,
    assistant_text: str = "thinking...",
    tool_name: str = "read_file",
    tool_args: str = '{"path": "/tmp/x"}',
    tool_result: str = "data",
    with_tool: bool = True,
    start_ts: float | None = None,
) -> list[dict[str, Any]]:
    """One user→assistant(+tool)→tool exchange as live host rows (uid + timestamp)."""
    user_text = user_text if user_text is not None else f"task {index}"
    base = 1_700_000_000.0 + index * 120.0 if start_ts is None else start_ts
    def stamp(offset: float) -> dict[str, Any]:
        return {"timestamp": base + offset}
    rows: list[dict[str, Any]] = [
        {
            "role": "user", "content": user_text,
            "message_uid": f"u-{index}-0", **stamp(0.0),
        },
        {
            "role": "assistant", "content": assistant_text,
            "message_uid": f"u-{index}-1", **stamp(30.0),
        },
    ]
    if with_tool:
        rows[-1]["tool_calls"] = [
            {
                "id": f"call_{index}", "type": "function",
                "function": {"name": tool_name, "arguments": tool_args},
            }
        ]
        rows.append(
            {
                "role": "tool", "tool_call_id": f"call_{index}", "content": tool_result,
                "message_uid": f"u-{index}-2", **stamp(35.0),
            }
        )
    return rows


def strip_for_request(msg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in msg.items() if k not in PERSISTENCE_FIELDS}


def build_request(
    conversation: list[dict[str, Any]], system_prompt: str = SYSTEM_PROMPT
) -> list[dict[str, Any]]:
    request: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    request.extend(strip_for_request(dict(m)) for m in conversation)
    return request


def fingerprint_list(messages: list[Any]) -> str:
    return json.dumps(messages, ensure_ascii=False, sort_keys=True, default=str)
