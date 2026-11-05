"""Unit identification — the identity layer between Hermes messages and HCE tags.

The conversation list (deep-cloned, full fields) is the IDENTITY source: units are
found and keyed here; the request copy is only ever written through the index
mapping verified in render.py.

Identity rules:
- A message's identity is ``u:<message_uid>`` when the durable uid is present
  (it survives in-place compaction, rotation and re-issues), else ``h:<sha1>``
  of its provider-visible shape. Identical uid-less messages deliberately share
  one identity: tags then apply to every occurrence, which is deterministic and
  semantically harmless for interchangeable content.
- Tool arcs are keyed by ``(owner identity, call_id)`` because provider call ids
  repeat across assistant turns. A result pairs with the EARLIEST unpaired call
  carrying the same call_id; results whose call vanished (compacted owner) are
  standalone arcs with ``owner=None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from agent.message_metadata import MESSAGE_UID, TOOL_CALL_UID, TOOL_CALL_UIDS

from .estimate import (
    IMAGE_PART_TOKENS,
    args_string_bytes,
    content_fingerprint,
    estimate_message_tokens,
)

FILE_PART_TYPES = frozenset({"image_url", "image", "file", "input_image"})


@dataclass
class Unit:
    """One recyclable unit found in the conversation."""

    kind: str                    # 'tool' | 'file'
    carrier_index: int           # conversation index of the message carrying the payload
    carrier_id: str
    uid: Optional[str]           # carrier uid when present
    content_hash: str
    token_count: int             # tokens of the droppable payload
    owner_id: Optional[str] = None    # tool only
    owner_index: Optional[int] = None  # tool only: conversation index of the assistant carrier
    owner_uid: Optional[str] = None
    call_id: Optional[str] = None
    call_name: Optional[str] = None
    args_bytes: Optional[int] = None
    part_index: Optional[int] = None  # file only
    result_is_last: bool = False      # tool only: result is the conversation's final message
    owner_has_reasoning: bool = False  # tool only: owner assistant replays reasoning (signature)


@dataclass
class ScanResult:
    units: list[Unit] = field(default_factory=list)
    identities: list[str] = field(default_factory=list)      # per conversation index
    uids: list[Optional[str]] = field(default_factory=list)
    hashes: list[str] = field(default_factory=list)
    timestamps: list[Optional[float]] = field(default_factory=list)
    fingerprints: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    assistant_reasoning: list[bool] = field(default_factory=list)
    token_estimates: list[int] = field(default_factory=list)


def parse_timestamp(value: Any) -> Optional[float]:
    """Tolerant timestamp parse (epoch float/int or ISO-8601 string); None on failure."""
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    if isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            return float(text)
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def message_identity(msg: dict[str, Any]) -> tuple[str, Optional[str], str]:
    """Return ``(identity, uid, content_hash)`` for one conversation message."""
    uid = msg.get(MESSAGE_UID)
    content_hash = content_fingerprint(msg)
    if uid:
        return f"u:{uid}", str(uid), content_hash
    return f"h:{content_hash}", None, content_hash


def _iter_calls(msg: dict[str, Any]) -> list[dict[str, Any]]:
    calls = msg.get("tool_calls")
    if not isinstance(calls, list):
        return []
    return [c for c in calls if isinstance(c, dict)]


def scan_conversation(conversation: list[dict[str, Any]]) -> ScanResult:
    """Walk the conversation once: identities, timestamps and recyclable units."""
    scan = ScanResult()
    pending_calls: list[dict[str, Any]] = []  # unpaired calls in arrival order

    for idx, msg in enumerate(conversation):
        if not isinstance(msg, dict):
            # Defensive: non-dict rows cannot be managed; keep index alignment.
            scan.identities.append("")
            scan.uids.append(None)
            scan.hashes.append("")
            scan.timestamps.append(None)
            scan.fingerprints.append("")
            scan.roles.append("")
            scan.assistant_reasoning.append(False)
            scan.token_estimates.append(0)
            continue

        identity, uid, content_hash = message_identity(msg)
        scan.identities.append(identity)
        scan.uids.append(uid)
        scan.hashes.append(content_hash)
        scan.timestamps.append(parse_timestamp(msg.get("timestamp")))
        scan.fingerprints.append(content_fingerprint(msg))
        role = str(msg.get("role") or "")
        scan.roles.append(role)
        scan.assistant_reasoning.append(bool(msg.get("reasoning_content")))
        scan.token_estimates.append(estimate_message_tokens(msg))

        if role == "assistant":
            # Per-occurrence uid map: {provider call id: occurrence uid}. Provider ids
            # repeat across turns; the occurrence uid is the precise pairing key.
            occ_map = msg.get(TOOL_CALL_UIDS)
            occ_map = occ_map if isinstance(occ_map, dict) else {}
            for call in _iter_calls(msg):
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                provider_id = str(call.get("id") or "")
                pending_calls.append(
                    {
                        "owner_index": idx,
                        "owner_id": identity,
                        "owner_uid": uid,
                        "owner_has_reasoning": bool(msg.get("reasoning_content")),
                        "call_id": provider_id,
                        "occ_uid": str(occ_map.get(provider_id) or ""),
                        "call_name": str((fn or {}).get("name") or ""),
                        "args_bytes": args_string_bytes((fn or {}).get("arguments")),
                    }
                )
        elif role == "tool":
            occ_uid = str(msg.get(TOOL_CALL_UID) or "")
            provider_id = str(msg.get("tool_call_id") or "")
            owner = None
            for cand in pending_calls:
                if occ_uid and cand["occ_uid"] == occ_uid:
                    owner = cand
                    pending_calls.remove(cand)
                    break
            if owner is None:
                for cand in pending_calls:
                    if provider_id and cand["call_id"] == provider_id:
                        owner = cand
                        pending_calls.remove(cand)
                        break
            result_is_last = idx == len(conversation) - 1
            scan.units.append(
                Unit(
                    kind="tool",
                    carrier_index=idx,
                    carrier_id=identity,
                    uid=uid,
                    content_hash=content_hash,
                    token_count=scan.token_estimates[idx],
                    owner_id=owner["owner_id"] if owner else None,
                    owner_index=owner["owner_index"] if owner else None,
                    owner_uid=owner["owner_uid"] if owner else None,
                    call_id=(owner["call_id"] if owner else provider_id) or None,
                    call_name=owner["call_name"] if owner else None,
                    args_bytes=owner["args_bytes"] if owner else None,
                    result_is_last=result_is_last,
                    owner_has_reasoning=(
                        owner["owner_has_reasoning"] if owner else bool(msg.get("reasoning_content"))
                    ),
                )
            )

        # File/media parts in any message's content list are individually recyclable.
        content = msg.get("content")
        if isinstance(content, list):
            for part_index, part in enumerate(content):
                if isinstance(part, dict) and str(part.get("type") or "") in FILE_PART_TYPES:
                    # An image inside a tool result is reclaimed together with its arc;
                    # tag it anyway so the arc's removal can compact it cleanly.
                    scan.units.append(
                        Unit(
                            kind="file",
                            carrier_index=idx,
                            carrier_id=identity,
                            uid=uid,
                            content_hash=content_hash,
                            token_count=IMAGE_PART_TOKENS,
                            part_index=part_index,
                        )
                    )
    return scan


def tool_arc_seq_key(unit: Unit) -> tuple[int, int]:
    """Ordering key for 'newest N tool calls' style rules (persistent seq order)."""
    return (unit.carrier_index, unit.part_index or 0)
