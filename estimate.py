"""Single token estimator for HCE (design 9.13: one function, results persisted).

Heuristic character-class estimator. Its systematic drift against a given model's
real tokenizer is handled by ONE static ratio frozen per session (engine
``_estimate_ratio``, default 1.0 = trust the heuristic) — never learned from
``prompt_tokens``, which also prices the system prompt and tool schemas.
All lanes share this estimator; the render path never re-tokenizes (unit counts are
persisted at tag time and only read back).
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

# Flat price for image/file parts (host's rough estimator is deliberately byte-blind
# to images too; a flat constant keeps our sums stable across passes).
IMAGE_PART_TOKENS = 1500
# ASCII-heavy text averages ~3.8 chars/token across modern tokenizers; CJK averages
# close to 1 token/char. Any systematic drift is corrected by the engine's frozen
# per-session ratio, not here — this function stays deterministic and stateless.
ASCII_CHARS_PER_TOKEN = 3.8
_TOOL_CALL_OVERHEAD_TOKENS = 8

_CJK_RANGES = (
    (0x2E80, 0x2EFF), (0x3000, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF), (0x20000, 0x2A6DF),
)


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def estimate_text_tokens(text: str) -> int:
    """Deterministic heuristic token estimate for one string."""
    if not text:
        return 0
    cjk = 0
    other = 0
    for ch in text:
        if _is_cjk(ch):
            cjk += 1
        else:
            other += 1
    return cjk + math.ceil(other / ASCII_CHARS_PER_TOKEN)


def _part_tokens(part: Any) -> int:
    if isinstance(part, str):
        return estimate_text_tokens(part)
    if not isinstance(part, dict):
        return estimate_text_tokens(str(part))
    ptype = part.get("type")
    if ptype in ("image_url", "image", "file", "input_image"):
        return IMAGE_PART_TOKENS
    if ptype == "text":
        return estimate_text_tokens(str(part.get("text") or ""))
    if ptype == "tool_use":
        return _tool_args_tokens(part.get("input")) + _TOOL_CALL_OVERHEAD_TOKENS
    if ptype == "tool_result":
        return _content_tokens(part.get("content"))
    # Unknown structured part: estimate its JSON serialization deterministically.
    return estimate_text_tokens(json.dumps(part, ensure_ascii=False, sort_keys=True, default=str))


def _content_tokens(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return estimate_text_tokens(content)
    if isinstance(content, list):
        return sum(_part_tokens(p) for p in content)
    return estimate_text_tokens(json.dumps(content, ensure_ascii=False, sort_keys=True, default=str))


def _tool_args_tokens(args: Any) -> int:
    """Token estimate for tool-call arguments (string values only, keys/numbers free)."""
    if args is None:
        return 0
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (TypeError, ValueError):
            return estimate_text_tokens(args)
    return _string_bytes_tokens(args)


def _string_bytes_tokens(obj: Any) -> int:
    total_bytes = 0
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, str):
            total_bytes += len(cur.encode("utf-8", errors="replace"))
        elif isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, (list, tuple)):
            stack.extend(cur)
    return math.ceil(total_bytes / ASCII_CHARS_PER_TOKEN)


def args_string_bytes(args: Any) -> int:
    """UTF-8 byte sum of every string VALUE in tool-call arguments (design 3.2).

    Keys, numbers, booleans and nulls count nothing. This is the shared 'small
    input' predicate every lane must use — never a second copy of the logic.
    """
    if args is None:
        return 0
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (TypeError, ValueError):
            return len(args.encode("utf-8", errors="replace"))
    total = 0
    stack = [args]
    while stack:
        cur = stack.pop()
        if isinstance(cur, str):
            total += len(cur.encode("utf-8", errors="replace"))
        elif isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, (list, tuple)):
            stack.extend(cur)
    return total


def estimate_tail_tokens(msg: dict[str, Any]) -> int:
    """Estimate for one message's content and tool inputs — reasoning excluded.

    The nudge measures tail mass (``T``) as what the agent could be asked to drop:
    text, tool inputs and results, file parts. Reasoning and signatures are excluded
    from both ``U`` and ``T`` in the design, because they are neither reclaimable nor
    content the model chose to keep — counting them would inflate the denominator and
    silently flatten every severity band.
    """
    total = _content_tokens(msg.get("content"))
    calls = msg.get("tool_calls")
    if isinstance(calls, list):
        for call in calls:
            fn = call.get("function") if isinstance(call, dict) else None
            if isinstance(fn, dict):
                total += _tool_args_tokens(fn.get("arguments")) + _TOOL_CALL_OVERHEAD_TOKENS
                total += estimate_text_tokens(str(fn.get("name") or ""))
    return total


def estimate_reasoning_tokens(msg: dict[str, Any]) -> int:
    """Tokens of the replayed reasoning on one message (0 when absent)."""
    reasoning = msg.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        return estimate_text_tokens(reasoning)
    return 0


def estimate_message_tokens(msg: dict[str, Any]) -> int:
    """Estimate for one message as the provider sees it (content + tool_calls +
    reasoning when replayed). Persistence-only fields are priced at zero because
    they never reach the wire."""
    return estimate_tail_tokens(msg) + estimate_reasoning_tokens(msg)


def content_fingerprint(msg: dict[str, Any]) -> str:
    """Stable short fingerprint of a message's provider-visible shape.

    Used to verify the request↔conversation 1:1 mapping and LKG head stability.
    Deliberately excludes persistence-only fields (uid/timestamp) and
    reasoning_content (replay is route-dependent; identity must not depend on it).
    """
    basis = {
        "role": msg.get("role"),
        "content": msg.get("content"),
        "tool_calls": msg.get("tool_calls"),
        "tool_call_id": msg.get("tool_call_id"),
        "name": msg.get("name"),
    }
    blob = json.dumps(basis, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()
