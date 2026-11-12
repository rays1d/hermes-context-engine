"""Request-copy rendering — the only place bytes are written.

Pure derivation: every output message is recomputed each pass from the request
base plus persisted state (tag rows + frozen markers). Nothing we emit is ever
persisted back into the transcript, so the same state always yields the same
bytes (invariants 1/5).

The render is COPY-based: the host's request list is never mutated. If the
render fails midway the host's list is still pristine — returning None then
leaves the request exactly as the host assembled it (fail-open seam). Each
message's content is derived exactly once from its base (markers, tag prefixes
and file-part transforms compose in one place, so part indexes never shift
under each other).

Mapping contract (verified before any write): ``request`` = head messages
(system prompt, prefill, ...) followed by exactly one rendered image per
conversation message, in order (``build_api_messages`` is 1:1 by construction;
turn_request_assembly inserts prefill right after the system prompt). The
verification is structural — length, role sequence, tool-call identities — plus
content fingerprints wherever they are predictable (the ``api_content`` sidecar
pins historical bytes). On any mismatch the render fails closed: the engine
returns None and the host sends the request unchanged.
"""

from __future__ import annotations

from typing import Any, Optional

from .nudge import NUDGE_OPEN, strip_reminders
from .units import FILE_PART_TYPES, ScanResult, Unit

PLACEHOLDER = "[dropped §{n}§]"
TAG_PREFIX = "§{n}§"
# Below this gap no marker is emitted at all: a 0m/1m marker on every quick exchange
# is pure noise (and pure tokens). Matches magic-context's temporal threshold.
MARKER_MIN_GAP_SECONDS = 300.0
MAX_TOLERATED_CONTENT_DRIFT = 3


class RenderMismatch(Exception):
    """Request↔conversation structural alignment failed; fail-open to the host."""


def verify_mapping(
    request: list[dict[str, Any]], conversation: list[dict[str, Any]], scan: ScanResult
) -> int:
    """Return head_len (number of leading non-history messages), or raise RenderMismatch."""
    conversation_len = len(scan.identities)
    head_len = len(request) - conversation_len
    if head_len < 0:
        raise RenderMismatch(
            f"request shorter than conversation ({len(request)} < {conversation_len})"
        )
    tolerated = 0
    for i in range(conversation_len):
        req = request[head_len + i]
        if not isinstance(req, dict):
            raise RenderMismatch(f"request[{head_len + i}] is not a dict")
        role = scan.roles[i]
        if str(req.get("role") or "") != role:
            raise RenderMismatch(
                f"role mismatch at conversation[{i}]: {req.get('role')!r} != {role!r}"
            )
        conv_msg = conversation[i] if isinstance(conversation[i], dict) else {}
        if role == "tool":
            req_call = str(req.get("tool_call_id") or "")
            conv_call = str(conv_msg.get("tool_call_id") or "")
            if req_call != conv_call:
                raise RenderMismatch(f"tool_call_id mismatch at conversation[{i}]")
        if role == "assistant" and not _same_call_ids(req.get("tool_calls"), conv_msg.get("tool_calls")):
            raise RenderMismatch(f"tool_calls mismatch at conversation[{i}]")
        # Content: strict whenever the sidecar pins the exact wire bytes; tolerated
        # otherwise (live composition, empty-payload healing happen at request time).
        sidecar = conv_msg.get("api_content")
        if isinstance(sidecar, str) and sidecar:
            if req.get("content") != sidecar:
                raise RenderMismatch(f"sidecar content drift at conversation[{i}]")
        elif req.get("content") != conv_msg.get("content"):
            tolerated += 1
            if tolerated > MAX_TOLERATED_CONTENT_DRIFT:
                raise RenderMismatch(
                    f"content drift at {tolerated} positions without sidecars; refusing to map"
                )
    return head_len


def _same_call_ids(req_calls: Any, conv_calls: Any) -> bool:
    req_ids = [str(c.get("id") or "") for c in req_calls] if isinstance(req_calls, list) else None
    conv_ids = [str(c.get("id") or "") for c in conv_calls] if isinstance(conv_calls, list) else None
    if req_ids is None or conv_ids is None:
        return req_ids == conv_ids
    return req_ids == conv_ids


def _append_text(content: Any, text: str) -> Any:
    """Append a block to string or parts-list content."""
    if isinstance(content, list):
        return list(content) + [{"type": "text", "text": text}]
    base = content if isinstance(content, str) else ("" if content is None else str(content))
    return f"{base}\n\n{text}" if base else text


def _strip_from_parts(content: Any) -> Any:
    if not isinstance(content, list) or NUDGE_OPEN not in repr(content):
        return content
    cleaned: list[Any] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
            text = strip_reminders(part["text"])
            if text.strip():
                cleaned.append({**part, "text": text})
            continue
        cleaned.append(part)
    return cleaned


def marker_text_for(gap_seconds: Optional[float]) -> str:
    """Deterministic elapsed-time marker, or empty string when nothing is emitted.

    Coarse buckets (m/h/d/w) keep the byte shape stable while still conveying scale.
    The DECISION is frozen at the message's first appearance — an empty marker is a
    decision too (design: 变更判定冻结). Never defer it to a later pass.
    """
    if gap_seconds is None or gap_seconds < MARKER_MIN_GAP_SECONDS:
        return ""
    total = int(gap_seconds)
    if total < 3600:
        return f"<!-- +{total // 60}m -->"
    if total < 86400:
        hours, minutes = total // 3600, (total % 3600) // 60
        return f"<!-- +{hours}h -->" if minutes == 0 else f"<!-- +{hours}h {minutes}m -->"
    if total < 7 * 86400:
        days, hours = total // 86400, (total % 86400) // 3600
        return f"<!-- +{days}d -->" if hours == 0 else f"<!-- +{days}d {hours}h -->"
    weeks, days = total // (7 * 86400), (total % (7 * 86400)) // 86400
    return f"<!-- +{weeks}w -->" if days == 0 else f"<!-- +{weeks}w {days}d -->"


def derive_content(
    base: Any,
    *,
    marker: Optional[str] = None,
    prefix_n: Optional[int] = None,
    placeholder_n: Optional[int] = None,
    file_ops: Optional[dict[int, tuple[int, dict[str, Any]]]] = None,
    prefix_active: bool = True,
) -> Any:
    """One deterministic derivation per message. Returns the BASE object itself
    when nothing applies (so untouched messages keep object identity).

    ``prefix_active=False`` (compress mode) suppresses every additive marker —
    placeholders still apply, but §N§ prefixes are not (re-)written into a
    transcript that is about to be committed as durable history."""
    if placeholder_n is not None:
        return PLACEHOLDER.format(n=placeholder_n)
    file_changed = bool(file_ops) and any(
        row.get("status") == "dropped" for _n, row in file_ops.values()
    )
    if not prefix_active and not file_changed:
        return base
    if not marker and prefix_n is None and not file_ops:
        return base
    if isinstance(base, list):
        parts: list[Any] = []
        if prefix_n is not None and prefix_active:
            parts.append({"type": "text", "text": TAG_PREFIX.format(n=prefix_n)})
        if marker:
            parts.append({"type": "text", "text": marker})
        for pidx, part in enumerate(base):
            hit = (file_ops or {}).get(pidx)
            if hit is not None:
                number, row = hit
                status = row.get("status")
                if status == "dropped":
                    parts.append({"type": "text", "text": PLACEHOLDER.format(n=number)})
                    continue
                if status == "active" and prefix_active:
                    parts.append(part)
                    parts.append({"type": "text", "text": TAG_PREFIX.format(n=number)})
                    continue
            parts.append(part)
        return parts
    text = base if isinstance(base, str) else ("" if base is None else str(base))
    marker_str = f"{marker}\n" if marker else ""
    prefix_str = f"{TAG_PREFIX.format(n=prefix_n)} " if (prefix_n is not None and prefix_active) else ""
    return f"{marker_str}{prefix_str}{text}"


def render_request(
    request: list[dict[str, Any]],
    conversation: list[dict[str, Any]],
    scan: ScanResult,
    unit_tag_numbers: dict[tuple, int],
    tag_rows: dict[int, dict[str, Any]],
    markers: dict[str, str],
    *,
    include_markers: bool = True,
    prefix_active: bool = True,
    skip_verification: bool = False,
    nudge: Optional[str] = None,
    frozen_nudges: Optional[dict[str, str]] = None,
    strip_stale_reminders: bool = False,
) -> tuple[list[Any], list[tuple[str, str]], int]:
    """Render the request copy. Returns ``(output, new_markers, head_len)``.

    ``unit_tag_numbers`` maps ``(kind, carrier_id, call_id, part_index)`` to the
    persisted tag number; ``tag_rows`` maps tag number to its state row.
    Raises RenderMismatch when the mapping cannot be verified — before any work.

    ``include_markers=False`` / ``prefix_active=False`` (compress mode) suppress
    additive bytes so the committed transcript stays clean of engine cosmetics.
    """
    head_len = len(request) - len(scan.identities)
    if not skip_verification:
        head_len = verify_mapping(request, conversation, scan)
    if head_len < 0:
        raise RenderMismatch(
            f"request shorter than conversation ({len(request)} < {len(scan.identities)})"
        )
    out: list[Any] = list(request)
    new_markers: list[tuple[str, str]] = []

    tool_by_carrier: dict[int, tuple[int, dict[str, Any], Unit]] = {}
    files_by_carrier: dict[int, dict[int, tuple[int, dict[str, Any]]]] = {}
    for unit in scan.units:
        key = (unit.kind, unit.carrier_id, unit.call_id, unit.part_index)
        number = unit_tag_numbers.get(key)
        if number is None:
            continue
        row = tag_rows.get(number)
        if row is None:
            continue
        if unit.kind == "tool":
            tool_by_carrier[unit.carrier_index] = (number, row, unit)
        else:
            files_by_carrier.setdefault(unit.carrier_index, {})[unit.part_index or 0] = (
                number,
                row,
            )

    request_removals: set[int] = set()             # request indexes to delete
    owner_call_strips: dict[int, set[str]] = set() # conversation indexes whose tool_calls lose calls

    for i in range(len(scan.identities)):
        role = scan.roles[i]
        req_index = head_len + i
        base_req = request[req_index]
        if not isinstance(base_req, dict):  # verified earlier; keep the type checker honest
            continue
        base_content = base_req.get("content")
        if strip_stale_reminders and (
            (isinstance(base_content, str) and NUDGE_OPEN in base_content)
            or isinstance(base_content, list)
        ):
            # Stale reminders may only be rewritten on passes that were going to change
            # historical bytes anyway; a derived pass replays them verbatim.
            base_content = (
                strip_reminders(base_content) if isinstance(base_content, str)
                else _strip_from_parts(base_content)
            )
        identity = scan.identities[i]

        # Reminders already frozen onto this message replay byte-exact; a message that
        # was the tail when it got one must keep it after the tail moves past it.
        frozen_nudge = (frozen_nudges or {}).get(identity)
        content_touched = False
        if frozen_nudge:
            base_content = _append_text(base_content, frozen_nudge)
            content_touched = True

        marker_text: Optional[str] = None
        if role == "user" and include_markers:
            frozen = markers.get(identity)
            if frozen is not None:
                marker_text = frozen or None
            else:
                anchor_ts = _previous_timestamp(scan, i)
                user_ts = scan.timestamps[i]
                gap = user_ts - anchor_ts if (anchor_ts is not None and user_ts is not None) else None
                text = marker_text_for(gap)
                new_markers.append((identity, text))
                marker_text = text or None

        placeholder_n: Optional[int] = None
        prefix_n: Optional[int] = None
        if role == "tool" and i in tool_by_carrier:
            number, row, unit = tool_by_carrier[i]
            status = row.get("status")
            if status == "dropped":
                if row.get("drop_form") == "remove":
                    request_removals.add(req_index)
                    if unit.owner_index is not None and unit.call_id:
                        owner_call_strips.add(unit.owner_index)
                    continue  # content irrelevant; the message is deleted below
                placeholder_n = number
            elif status == "active":
                prefix_n = number

        file_ops: dict[int, tuple[int, dict[str, Any]]] = {}
        if isinstance(base_content, list) and i in files_by_carrier:
            for pidx, part in enumerate(base_content):
                if not isinstance(part, dict) or str(part.get("type") or "") not in FILE_PART_TYPES:
                    continue
                hit = files_by_carrier[i].get(pidx)
                if hit is not None:
                    file_ops[pidx] = hit

        derived = derive_content(
            base_content, marker=marker_text, prefix_n=prefix_n,
            placeholder_n=placeholder_n, file_ops=file_ops or None,
            prefix_active=prefix_active,
        )
        if content_touched or derived is not base_content:
            copy = dict(base_req)
            copy["content"] = derived
            out[req_index] = copy

    # -- removal surgery ---------------------------------------------------------
    # Strip calls from owner assistants first (no index changes), collecting the
    # owners that emptied out; then delete messages highest-index-first.
    emptied_owners: set[int] = set()
    for conv_index in owner_call_strips:
        owner_index = head_len + conv_index
        if not (0 <= owner_index < len(out)):
            continue
        owner = out[owner_index]
        if not isinstance(owner, dict):
            continue
        calls = owner.get("tool_calls")
        if not isinstance(calls, list):
            continue
        removed_ids = {
            unit.call_id
            for unit in scan.units
            if unit.kind == "tool" and unit.owner_index == conv_index
            and _is_removed(tag_rows, unit_tag_numbers, unit)
        }
        kept = [c for c in calls if isinstance(c, dict) and str(c.get("id") or "") not in removed_ids]
        if len(kept) == len(calls):
            continue
        owner = dict(owner)
        out[owner_index] = owner
        if kept:
            owner["tool_calls"] = kept
        else:
            content = owner.get("content")
            has_text = bool(content) and not (isinstance(content, str) and not content.strip())
            if has_text:
                owner.pop("tool_calls", None)
            else:
                emptied_owners.add(owner_index)
    for req_index in sorted(request_removals | emptied_owners, reverse=True):
        if 0 <= req_index < len(out):
            del out[req_index]

    if nudge:
        # Tail-only: the reminder rides the newest user message, so it can never move a
        # cached prefix. Frozen history keeps whatever it had until a rebuild strips it.
        for rev in range(len(out) - 1, -1, -1):
            msg = out[rev]
            if isinstance(msg, dict) and msg.get("role") == "user":
                out[rev] = {**msg, "content": _append_text(msg.get("content"), nudge)}
                break
    return out, new_markers, head_len


def _is_removed(
    tag_rows: dict[int, dict[str, Any]], unit_tag_numbers: dict[tuple, int], unit: Unit
) -> bool:
    number = unit_tag_numbers.get((unit.kind, unit.carrier_id, unit.call_id, unit.part_index))
    if number is None:
        return False
    row = tag_rows.get(number)
    return bool(row and row.get("status") == "dropped" and row.get("drop_form") == "remove")


def _previous_timestamp(scan: ScanResult, index: int) -> Optional[float]:
    """Timestamp of the immediately preceding message (any role).

    The previous message — not the previous *assistant* reply — is the right right-hand
    endpoint: it is when the user last saw something happen, so tool round-trips in
    between are counted in the elapsed gap instead of being silently excluded.
    """
    for j in range(index - 1, -1, -1):
        if scan.timestamps[j] is not None:
            return scan.timestamps[j]
    return None
