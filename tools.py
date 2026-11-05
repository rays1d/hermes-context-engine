"""ctx_reduce / ctx_expand — tool schemas and response copy contracts.

The tool description is protocol, not documentation (design 4.2): it must say
plainly that stamping is not deletion, that extra calls cost nothing, and every
error must name the next action. The four response semantics (stamped / already
queued / nothing to reclaim / refused) get four distinct copies — a fuzzy
"success" makes models retry blindly (design 9.15).
"""

from __future__ import annotations

import json
from typing import Any

CTX_REDUCE_SCHEMA = {
    "name": "ctx_reduce",
    "description": (
        "Stamp context tags so their content leaves your working context. This is NOT "
        "deletion: stamping QUEUES the tag — the content drops out of the request at a "
        "later turn boundary, and the original stays retrievable via ctx_expand at any "
        "time. Look at each tag before stamping; many small targeted stamps beat one "
        "sweep, and never blanket-stamp a wide range. Tag ids are the §N§ markers shown "
        "on tool results and file parts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "tags": {
                "type": "string",
                "description": 'Tag ids to stamp, single or range form, e.g. "3-5" or "1,2,9".',
            },
        },
        "required": ["tags"],
    },
}

CTX_EXPAND_SCHEMA = {
    "name": "ctx_expand",
    "description": (
        "Retrieve the original content behind stamped tags (ctx_reduce). Use it whenever "
        "you need the full text of something that already left the context — dropped tool "
        "results, file parts, or their arguments. Works even long after the drop."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "tags": {
                "type": "string",
                "description": 'Tag id(s) to retrieve, e.g. "7" or "3-5".',
            },
        },
        "required": ["tags"],
    },
}


def tool_schemas() -> list[dict[str, Any]]:
    return [CTX_REDUCE_SCHEMA, CTX_EXPAND_SCHEMA]


def parse_tag_ids(raw: Any) -> tuple[list[int], list[str]]:
    """Parse ``"3-5"``, ``"1,2,9"``, ``"§3§"``, or a list of the same.

    Returns ``(ids_in_input_order_deduplicated, unparsed_tokens)``.
    """
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        n = int(raw)
        return ([n] if n > 0 else []), ([] if n > 0 else [str(raw)])
    if isinstance(raw, list):
        ids: list[int] = []
        bad: list[str] = []
        for item in raw:
            sub_ids, sub_bad = parse_tag_ids(item)
            ids.extend(sub_ids)
            bad.extend(sub_bad)
        return _dedupe(ids), bad
    if not isinstance(raw, str) or not raw.strip():
        return [], ([] if raw is None else [str(raw)])
    ids = []
    bad = []
    for token in raw.replace("，", ",").replace("、", ",").split(","):
        token = token.strip().replace("§", "").strip()
        if not token:
            continue
        if "-" in token:
            bounds = token.split("-", 1)
            try:
                lo, hi = int(bounds[0].strip()), int(bounds[1].strip())
            except ValueError:
                bad.append(token)
                continue
            if lo > hi or lo <= 0:
                bad.append(token)
                continue
            ids.extend(range(lo, hi + 1))
        else:
            try:
                n = int(token)
            except ValueError:
                bad.append(token)
                continue
            if n <= 0:
                bad.append(token)
                continue
            ids.append(n)
    return _dedupe(ids), bad


def _dedupe(ids: list[int]) -> list[int]:
    seen: set[int] = set()
    out = []
    for n in ids:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def reduce_response(
    stamped: list[int],
    held: list[int],
    already: list[int],
    skipped_self: list[int],
    refused: list[int],
    unknown: list[int],
    max_tag: int,
) -> str:
    """Four distinct semantics; every refusal names the next action."""
    parts: list[str] = []
    if stamped or held:
        queued = stamped + held
        names = ", ".join(f"§{n}§" for n in queued)
        parts.append(
            f"Stamped {len(queued)} tag(s): {names}. They are queued, not deleted — the "
            "content leaves the request at an upcoming turn boundary and originals stay "
            "retrievable via ctx_expand."
        )
        if held:
            held_names = ", ".join(f"§{n}§" for n in held)
            parts.append(
                f"Recently used, so {held_names} stay visible for now and drop later "
                "automatically — no further action needed."
            )
    if already:
        names = ", ".join(f"§{n}§" for n in already)
        parts.append(f"No new action is needed for {names} — already stamped or dropped.")
    if skipped_self:
        parts.append("ctx_reduce's own old calls clean up automatically; no stamping needed.")
    if refused:
        names = ", ".join(f"§{n}§" for n in refused)
        parts.append(
            f"Refused: {names} sit before a compression boundary and cannot be re-stamped. "
            "Use ctx_expand if you need their content."
        )
    if unknown:
        names = ", ".join(f"§{n}§" for n in unknown)
        parts.append(
            f"Tag(s) {names} do not exist. Valid tags run §1§–§{max_tag}§; untagged content "
            "has no marker and cannot be stamped."
        )
    if not parts:
        parts.append("Nothing to reclaim: no matching tags were given.")
    return json.dumps(
        {
            "stamped": stamped,
            "held": held,
            "already": already,
            "skipped_self": skipped_self,
            "refused": refused,
            "unknown": unknown,
            "summary": " ".join(parts),
        },
        ensure_ascii=False,
    )


def expand_response(sections: list[tuple[int, str]], notes: list[str]) -> str:
    """Originals per tag; unknown tags point at the valid range instead of failing vague."""
    payload: dict[str, Any] = {"content": "", "sections": []}
    body_parts = []
    if sections:
        chunks = []
        for number, text in sections:
            chunks.append(f"=== §{number}§ ===\n{text}")
        body = "\n\n".join(chunks)
        payload["content"] = body
        payload["sections"] = [{"tag": n, "content": t} for n, t in sections]
        body_parts.append(f"Retrieved {len(sections)} tag(s).")
    else:
        body_parts.append("No stored original for the requested tags.")
    if notes:
        body_parts.extend(notes)
        payload["notes"] = notes
    payload["summary"] = " ".join(body_parts)
    return json.dumps(payload, ensure_ascii=False)
