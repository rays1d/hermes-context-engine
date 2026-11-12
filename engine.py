"""HermesContextEngine — cache-stable context management for Hermes (M0).

Discipline map (design doc):
- 引擎永不抛出：select_context/compress 全身 try/except；内部故障 → LKG 重放。
- 决定/执行分离：ctx_reduce 只写 pending_ops；落地只发生在有重建许可的 pass。
- 单事务：一次 pass 的全部状态写入 + 渲染读态在同一 BEGIN IMMEDIATE 里。
- 存储即真相：gateway 每轮重建引擎实例，跨 pass 状态一律按 session_id 落库。
- real-or-absent：工具参数永远是真实值或整体移除，绝不写入合成值。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from agent.context_engine import ContextEngine

from .estimate import content_fingerprint, estimate_message_tokens
from .nudge import decide as nudge_decide
from .render import PLACEHOLDER, RenderMismatch, render_request
from .store import SchemaFenceError, Store
from .tools import expand_response, parse_tag_ids, reduce_response, tool_schemas
from .units import Unit, scan_conversation

logger = logging.getLogger("hermes.context_engine")

SMALL_ARGS_BYTES = 1024          # design 3.2 骨架"小输入"阈值（全车道共享同一函数）
RECENT_CALL_WINDOW = 20          # 最新 N 次工具调用
STRUCTURAL_TOOL_MINIMUM = 3      # 最新 N 个工具弧永远保留（眼睛里得有工具调用长什么样）
PROTECTED_TOOL_KEEP = {"ctx_reduce": 3}   # todo_list 探测留待后续里程碑
QUEUE_REBUILD_FRACTION = 0.02    # 自行重建门槛：排队 ≥ 2% **原始窗口**
HELD_PASS_LIMIT = 3              # 连续第 N 趟被挂住也放行（否则小会话永远等不到 2%）
PROTECTION_FRACTION = 0.05
PROTECTION_FLOOR_FRAC = 0.08
PROTECTION_FLOOR_ABS = 16_000
PROTECTION_CEILING = 64_000
MIN_RATIO = 0.05
MAX_RATIO = 50.0
_PLACEHOLDER_RE = re.compile(r"^\[dropped §(\d+)§\]$")


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        logger.warning("HCE config %s=%r is not a number; using default %s", key, raw, default)
        return default


def _env_duration(key: str, default: Optional[float]) -> Optional[float]:
    """Seconds, or ``"5m"``/``"1h"`` style, or ``"never"``/``0`` for off."""
    raw = os.environ.get(key)
    if raw is None or not str(raw).strip():
        return default
    text = str(raw).strip().lower()
    if text in {"never", "off", "0", "none"}:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    units = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
    multiplier = units.get(text[-1:])
    if multiplier:
        try:
            return float(text[:-1]) * multiplier
        except ValueError:
            pass
    logger.warning("HCE config %s=%r is not a duration; using default %s", key, raw, default)
    return default


def _last_user_identity(scan: Any) -> Optional[str]:
    """Identity of the newest user message (the only place a fresh reminder may ride)."""
    for i in range(len(scan.roles) - 1, -1, -1):
        if scan.roles[i] == "user" and scan.identities[i]:
            return scan.identities[i]
    return None


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class HermesContextEngine(ContextEngine):
    """M0: tag model + ctx_reduce/ctx_expand + landing lanes + LKG + time markers."""

    _ENGINE_NAME = "context-engine"

    def __init__(self) -> None:
        self._enabled = _env_bool("HCE_ENABLED", True)
        self._cache_ttl = _env_duration("HCE_CACHE_TTL", 300.0)
        self._threshold_pct = _env_float("HCE_EXECUTE_THRESHOLD", 65.0)
        self.threshold_percent = self._threshold_pct / 100.0
        self._store: Optional[Store] = None
        self._store_failed = False
        self._session_id: Optional[str] = None
        self._hermes_home: Optional[str] = None
        self._warned = False

    # ------------------------------------------------------------------
    # identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._ENGINE_NAME

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def bind_session_state(
        self, session_db: Any = None, session_id: Any = None, **kwargs: Any
    ) -> None:
        if session_id:
            self._session_id = str(session_id)

    def on_session_start(self, session_id: str, **kwargs: Any) -> None:
        try:
            self._session_id = str(session_id or "") or None
            home = kwargs.get("hermes_home")
            if home:
                self._hermes_home = str(home)
            store = self._open_store()
            if store is None or not self._session_id:
                return
            old_session_id = kwargs.get("old_session_id")
            boundary = kwargs.get("boundary_reason")
            with store.transaction() as conn:
                if boundary == "compression" and old_session_id and str(old_session_id) != self._session_id:
                    store.migrate_session(conn, str(old_session_id), self._session_id)
                    logger.info(
                        "HCE session rotated %s -> %s; state migrated",
                        old_session_id, self._session_id,
                    )
                store.set_meta(conn, self._session_id, "model", str(kwargs.get("model") or ""))
                store.set_meta(
                    conn, self._session_id, "context_length", int(self.context_length or 0)
                )
                # Reload usage so pass 1 of this turn classifies with real numbers
                # (the gateway rebuilds this engine every turn).
                persisted = store.get_meta_in_tx(conn, self._session_id, "last_prompt_tokens", 0)
                self.last_prompt_tokens = int(persisted or 0)
        except Exception:
            logger.warning("HCE on_session_start failed; continuing unbound", exc_info=True)

    def on_session_end(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self._session_id = None

    def _bump(self, conn: Any, key: str, by: int = 1) -> None:
        """Per-session counter in ``session_meta`` (cheap: same pass transaction)."""
        store, sid = self._store, self._session_id
        if store is None or not sid:
            return
        try:
            current = int(store.get_meta_in_tx(conn, sid, key, 0) or 0)
            store.set_meta(conn, sid, key, current + by)
        except Exception:
            logger.debug("HCE counter %s failed", key, exc_info=True)

    def _bump_safe(self, key: str, by: int = 1) -> None:
        """Counter bump outside a pass transaction (failure paths)."""
        store = self._store
        if store is None or not self._session_id:
            return
        try:
            with store.transaction() as conn:
                self._bump(conn, key, by)
        except Exception:
            logger.debug("HCE counter %s failed outside txn", key, exc_info=True)

    def update_model(
        self, model: str, context_length: int, base_url: str = "", api_key: str = "",
        provider: str = "", api_mode: str = "",
    ) -> None:
        """HCE owns its threshold policy outright (no per-model override table): the
        host-resolved context_length is the only input that matters here."""
        self.model = str(model or "")
        self.context_length = int(context_length or 0)
        self.threshold_percent = self._threshold_pct / 100.0
        self.threshold_tokens = int(self.context_length * self.threshold_percent)

    def update_from_response(self, usage: dict[str, Any]) -> None:
        """Track provider token counters for status only.

        Deliberately NO runtime estimate calibration: the estimator's drift is handled
        by a single static ratio frozen per session (``_estimate_ratio``). Learning a
        ratio from ``prompt_tokens`` would mix classes — that counter prices the whole
        request (system prompt + tool schemas + messages) while our estimate covers
        messages only, so the fixed overhead lands in the ratio as fake drift.
        """
        prompt = int((usage or {}).get("prompt_tokens") or 0)
        self.last_prompt_tokens = prompt
        self.last_completion_tokens = int((usage or {}).get("completion_tokens") or 0)
        self.last_total_tokens = int((usage or {}).get("total_tokens") or 0)
        store = self._store
        if store is None or not self._session_id or prompt <= 0:
            return
        try:
            store.write_meta(self._session_id, "last_prompt_tokens", prompt)
        except Exception:
            logger.debug("HCE usage persist failed", exc_info=True)

    # ------------------------------------------------------------------
    # host-driven compaction policy
    # ------------------------------------------------------------------

    def should_compress(self, prompt_tokens: int | None = None) -> bool:
        return False  # HCE owns rebuilds inside select_context; host triggers stay off

    def has_content_to_compress(self, messages: list[dict[str, Any]]) -> bool:
        try:
            if self._store is not None and self._session_id:
                if self._store.pending_token_sum_quick(self._session_id):
                    return True
            scan = scan_conversation(messages or [])
            return any(unit.kind == "tool" for unit in scan.units)
        except Exception:
            return True

    # ------------------------------------------------------------------
    # the pass (select_context)
    # ------------------------------------------------------------------

    def select_context(
        self,
        request_messages: list[dict[str, Any]],
        *,
        conversation_messages: list[dict[str, Any]] | None = None,
        incoming_message: dict[str, Any] | None = None,
        budget_tokens: int = 0,
    ) -> list[dict[str, Any]] | None:
        try:
            if not self._enabled:
                return None
            if not isinstance(request_messages, list) or not conversation_messages:
                return None
            if not self._session_id:
                self._warn_once("no bound session; pass skipped")
                return None
            store = self._open_store()
            if store is None:
                return self._lkg_replay(request_messages)
            now = time.time()
            with store.transaction() as conn:
                sid = self._session_id
                scan = scan_conversation(conversation_messages)
                unit_tag_numbers, tag_rows, changed = self._reconcile(
                    conn, scan, conversation_messages, now, allow_compaction=True
                )
                protected = self._protected_tag_numbers(conn, scan, unit_tag_numbers)
                pending = store.load_pending(conn, sid)
                queued_tokens = 0
                queued_count = 0
                for p in pending:
                    number = int(p["tag_number"])
                    if number in protected:
                        continue
                    queued_tokens += self._tag_token(tag_rows, number)
                    queued_count += 1
                reasons = self._rebuild_reasons(
                    conn, now, queued_tokens, queued_count, changed
                )
                landed = False
                if reasons:
                    landed = self._land_pending(
                        conn, conversation_messages, scan, unit_tag_numbers, tag_rows,
                        pending, protected, now,
                    )
                nudge_text = self._nudge_for_pass(
                    conn, scan, unit_tag_numbers, tag_rows, protected, landed
                )
                markers = store.load_markers(conn, sid)
                if reasons:
                    # A rebuilding pass may rewrite history: retire every stale reminder
                    # instead of replaying them, then let this pass decide a fresh one.
                    store.clear_nudges(conn, sid)
                    frozen_nudges: dict[str, str] = {}
                else:
                    frozen_nudges = store.load_nudges(conn, sid)
                output, new_markers, head_len = render_request(
                    request_messages, conversation_messages, scan,
                    unit_tag_numbers, tag_rows, markers,
                    nudge=nudge_text, frozen_nudges=frozen_nudges,
                    strip_stale_reminders=bool(reasons),
                )
                if nudge_text:
                    # Freeze immediately: once the tail moves past this message it must
                    # replay the exact same reminder bytes.
                    target = _last_user_identity(scan)
                    if target is not None:
                        store.freeze_nudge(conn, sid, target, nudge_text)
                for identity, text in new_markers:
                    store.freeze_marker(conn, sid, identity, text)
                est_total = sum(estimate_message_tokens(m) for m in output)
                store.set_meta(conn, sid, "last_est_total", int(est_total))
                store.set_meta(conn, sid, "last_request_at", now)
                self._bump(conn, "passes")
                if reasons:
                    self._bump(conn, "rebuilds")
                    self._bump(conn, "rebuild:" + reasons[0].split("(")[0])
                self._update_lkg(conn, output, head_len, scan)
            return output
        except RenderMismatch as exc:
            logger.warning("HCE mapping failed; sending unmodified request (%s)", exc)
            self._bump_safe("failopens")
            return None
        except Exception:
            logger.warning("HCE pass failed; attempting LKG replay", exc_info=True)
            self._bump_safe("failopens")
            replayed = self._lkg_replay(request_messages)
            if replayed is not None:
                self._bump_safe("lkg_replays")
            return replayed

    def _rebuild_reasons(
        self, conn: Any, now: float, queued_tokens: int, queued_count: int, changed: bool,
    ) -> list[str]:
        """重建许可：每趟 pass 算一次，所有车道共享（不变式 5）。M0 触发器：
        执行阈值、空闲超时、排队量占原始窗口 2%、连续第 3 趟挂住、对账变更。"""
        sid = self._session_id or ""
        store = self._store
        assert store is not None
        reasons: list[str] = []
        ctx_len = int(self.context_length or 0)
        usage_pct = (
            self.last_prompt_tokens / ctx_len * 100.0
            if ctx_len > 0 and self.last_prompt_tokens > 0 else 0.0
        )
        if usage_pct >= self._threshold_pct:
            reasons.append(f"threshold({usage_pct:.1f}%)")
        if self._cache_ttl is not None:
            last_req = store.get_meta_in_tx(conn, sid, "last_request_at", None)
            if last_req and (now - float(last_req)) >= self._cache_ttl:
                reasons.append("idle")
        if queued_tokens > 0 and ctx_len > 0:
            ratio = self._estimate_ratio(conn)
            # 2% of the RAW window (not of usable): the gate is about "is this worth a
            # bust", and a bigger window should need a bigger pile to justify one.
            if queued_tokens * ratio >= max(QUEUE_REBUILD_FRACTION * ctx_len, 1.0):
                reasons.append("queue")
        if queued_count > 0 and not reasons:
            held = int(store.get_meta_in_tx(conn, sid, "held_passes", 0) or 0) + 1
            store.set_meta(conn, sid, "held_passes", held)
            if held >= HELD_PASS_LIMIT:
                reasons.append(f"held({held})")
        if changed:
            # Compaction removed a tag that used to be on the wire: past bytes must change.
            reasons.append("reconcile")
        if reasons:
            store.set_meta(conn, sid, "held_passes", 0)
            logger.info("HCE rebuild permitted: %s (session=%s)", ",".join(reasons), sid)
        return reasons

    # ------------------------------------------------------------------
    # reconcile + landing
    # ------------------------------------------------------------------

    def _reconcile(
        self, conn: Any, scan: Any, conversation: list[dict[str, Any]],
        now: float, *, allow_compaction: bool,
    ) -> tuple[dict[tuple, int], dict[int, dict[str, Any]], bool]:
        """Match persisted tags to current units; assign numbers to new units; compact
        vanished tags. ``allow_compaction=False`` (compress on a partial head) never
        compacts: the unseen tail still exists in the real transcript."""
        sid = self._session_id or ""
        store = self._store
        assert store is not None
        rows = store.load_tags(conn, sid)
        tag_rows = {int(r["tag_number"]): dict(r) for r in rows}
        by_uid: dict[tuple, list[int]] = {}
        by_hash: dict[tuple, list[int]] = {}
        for number, row in tag_rows.items():
            kind = row["kind"]
            call = row["call_id"]
            part = row["part_index"]
            if row["uid"]:
                by_uid.setdefault((kind, row["uid"], call, part), []).append(number)
            by_hash.setdefault((kind, row["content_hash"], call, part), []).append(number)
        unit_tag_numbers: dict[tuple, int] = {}
        used: set[int] = set()
        next_number = store.next_tag_number(conn, sid)
        seq = max((int(r["seq"]) for r in tag_rows.values()), default=0)
        changed = False

        for unit in scan.units:
            key = (unit.kind, unit.carrier_id, unit.call_id, unit.part_index)
            pick = self._match_tag(unit, by_uid, by_hash, tag_rows, used)
            if pick is None and unit.kind == "tool":
                pick = self._match_placeholder(unit, conversation, tag_rows, used, conn, sid)
            if pick is not None:
                unit_tag_numbers[key] = pick
                used.add(pick)
                continue
            seq += 1
            number = next_number
            next_number += 1
            tag = {
                "session_id": sid, "tag_number": number, "kind": unit.kind,
                "carrier_id": unit.carrier_id, "uid": unit.uid,
                "content_hash": unit.content_hash, "owner_id": unit.owner_id,
                "call_id": unit.call_id, "call_name": unit.call_name,
                "args_bytes": unit.args_bytes, "part_index": unit.part_index,
                "seq": seq, "status": "active", "drop_form": None,
                "token_count": int(unit.token_count), "created_at": now,
            }
            store.insert_tag(conn, tag)
            tag_rows[number] = tag
            unit_tag_numbers[key] = number
            used.add(number)

        if allow_compaction:
            for number, row in list(tag_rows.items()):
                if number in used:
                    continue
                store.update_tag_status(conn, sid, number, "compacted")
                store.dequeue(conn, sid, number)
                del tag_rows[number]
                changed = True
            live_ids = set(scan.identities)
            stale = [mid for mid in store.load_markers(conn, sid) if mid not in live_ids]
            if stale:
                store.drop_markers(conn, sid, stale)
                changed = True
            stale_nudges = [mid for mid in store.load_nudges(conn, sid) if mid not in live_ids]
            if stale_nudges:
                store.drop_nudges(conn, sid, stale_nudges)
                changed = True
        return unit_tag_numbers, tag_rows, changed

    def _match_tag(
        self, unit: Unit,
        by_uid: dict[tuple, list[int]], by_hash: dict[tuple, list[int]],
        tag_rows: dict[int, dict[str, Any]], used: set[int],
    ) -> Optional[int]:
        if unit.uid:
            for number in by_uid.get((unit.kind, unit.uid, unit.call_id, unit.part_index), []):
                if number not in used:
                    return number
        for number in by_hash.get((unit.kind, unit.content_hash, unit.call_id, unit.part_index), []):
            row = tag_rows[number]
            # A uid-pinned tag only matches its own uid; hash-only tags follow content.
            if row["uid"] and row["uid"] != unit.uid:
                continue
            if number not in used:
                return number
        return None

    def _match_placeholder(
        self, unit: Unit, conversation: list[dict[str, Any]],
        tag_rows: dict[int, dict[str, Any]], used: set[int], conn: Any, sid: str,
    ) -> Optional[int]:
        """A committed placeholder carrier must not spawn a fresh tag: re-bind the
        dropped tag whose number the placeholder names (design 9.21 对账)."""
        if unit.carrier_index >= len(conversation):
            return None
        msg = conversation[unit.carrier_index]
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, str):
            return None
        match = _PLACEHOLDER_RE.match(content.strip())
        if not match:
            return None
        number = int(match.group(1))
        row = tag_rows.get(number)
        if row is None or row["kind"] != "tool" or number in used:
            return None
        if row.get("call_id") not in (None, unit.call_id):
            return None
        self._store.rebind_tag_carrier(
            conn, sid, number, unit.carrier_id, unit.uid, unit.content_hash
        )
        row["carrier_id"], row["uid"], row["content_hash"] = (
            unit.carrier_id, unit.uid, unit.content_hash
        )
        return number

    def _land_pending(
        self, conn: Any, conversation: list[dict[str, Any]], scan: Any,
        unit_tag_numbers: dict[tuple, int], tag_rows: dict[int, dict[str, Any]],
        pending: list[dict[str, Any]], protected: Optional[set[int]], now: float,
    ) -> bool:
        """落地三种形态：占位符（文件）、骨架（工具结果→占位符）、整体移除（弧消失）。
        Protected tags stay queued (held); compress passes protected=None to force-land."""
        sid = self._session_id or ""
        store = self._store
        assert store is not None
        if not pending:
            return False
        number_to_unit: dict[int, Unit] = {}
        for unit in scan.units:
            number = unit_tag_numbers.get(
                (unit.kind, unit.carrier_id, unit.call_id, unit.part_index)
            )
            if number is not None:
                number_to_unit[number] = unit
        landed = False
        for entry in pending:
            number = int(entry["tag_number"])
            row = tag_rows.get(number)
            if row is None or row.get("status") != "active":
                store.dequeue(conn, sid, number)
                continue
            if protected is not None and number in protected:
                continue  # held: accepted earlier, lands once it leaves the window
            form = row.get("drop_form") or "skeleton"
            unit = number_to_unit.get(number)
            if form == "remove" and (unit is None or self._removal_unsafe(unit, conversation)):
                form = "skeleton"  # land-time safety downgrade (design 3.2 三条例外)
            payload = self._source_payload(conversation, unit)
            if payload is not None:
                store.store_source(conn, sid, number, payload)
            store.update_tag_status(conn, sid, number, "dropped", drop_form=form)
            store.dequeue(conn, sid, number)
            tag_rows[number] = {**row, "status": "dropped", "drop_form": form}
            if form == "remove" and unit is not None and unit.kind == "tool":
                self._compact_inner_files(
                    conn, conversation, scan, unit_tag_numbers, tag_rows, unit
                )
            landed = True
        return landed

    def _nudge_for_pass(
        self, conn: Any, scan: Any, unit_tag_numbers: dict[tuple, int],
        tag_rows: dict[int, dict[str, Any]], protected: set[int], landed: bool,
    ) -> Optional[str]:
        """Decide whether a reduction reminder rides this pass (tail-only bytes)."""
        store = self._store
        sid = self._session_id or ""
        if store is None or not sid:
            return None
        usable = int(self.context_length or 0) * (self._threshold_pct / 100.0)
        count = 0
        tokens = 0
        oldest: list[tuple[int, Optional[str]]] = []
        for unit in scan.units:
            number = unit_tag_numbers.get(
                (unit.kind, unit.carrier_id, unit.call_id, unit.part_index)
            )
            if number is None or number in protected:
                continue
            row = tag_rows.get(number)
            if row is None or row.get("status") != "active":
                continue
            count += 1
            tokens += int(row.get("token_count") or 0)
            if len(oldest) < 4:
                oldest.append((number, row.get("call_name")))
        decision = nudge_decide(
            reclaimable_tokens=tokens,
            reclaimable_count=count,
            usable=usable,
            oldest=oldest,
            last_fired_tokens=store.get_meta_in_tx(conn, sid, "nudge_fired_tokens", None),
            last_level=store.get_meta_in_tx(conn, sid, "nudge_level", None),
            just_landed=landed,
        )
        if decision is None:
            return None
        level, text = decision
        store.set_meta(conn, sid, "nudge_fired_tokens", int(tokens))
        store.set_meta(conn, sid, "nudge_level", level)
        self._bump(conn, "nudges")
        self._bump(conn, f"nudge_{level}")
        logger.info(
            "HCE nudge %s: %d outputs / %d tokens reclaimable (session=%s)",
            level, count, tokens, sid,
        )
        return text

    def _removal_unsafe(self, unit: Unit, conversation: list[dict[str, Any]]) -> bool:
        """Forms that must degrade to a skeleton at land time.

        Removing a call whose result is the request tail, or that carries signed
        reasoning, or whose owner would be emptied (leaving a text-less assistant row
        or two assistant rows adjacent — the exact shape behind shipped provider 400s),
        keeps the call with its REAL arguments instead.
        """
        if unit.kind != "tool":
            return False
        if unit.result_is_last or unit.owner_has_reasoning:
            return True
        return self._owner_would_empty(conversation, unit)

    def _owner_would_empty(
        self, conversation: list[dict[str, Any]], unit: Unit,
    ) -> bool:
        if unit.owner_index is None or unit.owner_index >= len(conversation):
            return False
        owner = conversation[unit.owner_index]
        if not isinstance(owner, dict):
            return False
        content = owner.get("content")
        has_text = bool(content) and not (isinstance(content, str) and not content.strip())
        calls = owner.get("tool_calls")
        call_count = len(calls) if isinstance(calls, list) else 0
        return not has_text and call_count <= 1

    def _compact_inner_files(
        self, conn: Any, conversation: list[dict[str, Any]], scan: Any,
        unit_tag_numbers: dict[tuple, int], tag_rows: dict[int, dict[str, Any]],
        arc: Unit,
    ) -> None:
        sid = self._session_id or ""
        for unit in scan.units:
            if unit.kind != "file" or unit.carrier_id != arc.carrier_id:
                continue
            number = unit_tag_numbers.get(
                (unit.kind, unit.carrier_id, unit.call_id, unit.part_index)
            )
            if number is None:
                continue
            row = tag_rows.get(number)
            if row is None or row.get("status") != "active":
                continue
            payload = self._source_payload(conversation, unit)
            if payload is not None:
                self._store.store_source(conn, sid, number, payload)
            self._store.update_tag_status(conn, sid, number, "compacted")
            del tag_rows[number]

    def _source_payload(
        self, conversation: list[dict[str, Any]], unit: Optional[Unit],
    ) -> Optional[dict[str, Any]]:
        if unit is None or unit.carrier_index >= len(conversation):
            return None
        carrier = conversation[unit.carrier_index]
        if not isinstance(carrier, dict):
            return None
        if unit.kind == "tool":
            args_text: Any = None
            if unit.owner_index is not None and unit.owner_index < len(conversation):
                owner = conversation[unit.owner_index]
                calls = owner.get("tool_calls") if isinstance(owner, dict) else None
                if isinstance(calls, list):
                    for call in calls:
                        if isinstance(call, dict) and str(call.get("id") or "") == (unit.call_id or ""):
                            fn = call.get("function")
                            args_text = fn.get("arguments") if isinstance(fn, dict) else None
                            break
            return {
                "kind": "tool",
                "call_name": unit.call_name,
                "arguments": args_text,
                "result": carrier.get("content"),
            }
        if unit.kind == "file":
            content = carrier.get("content")
            if isinstance(content, list) and unit.part_index is not None and unit.part_index < len(content):
                return {"kind": "file", "part": content[unit.part_index]}
            return None
        return None

    # ------------------------------------------------------------------
    # protection window
    # ------------------------------------------------------------------

    def _estimate_ratio(self, conn: Any) -> float:
        """Provider-tokens-per-estimated-unit, FROZEN once per session.

        Deliberately static: the estimator is a deterministic char-class heuristic and
        its systematic drift does not change mid-session, so learning it per turn only
        adds state that can wander. Resolved once (session_meta) from
        ``HCE_ESTIMATE_RATIO`` (default 1.0 = trust the heuristic), clamped to sane
        bounds. Raise it when a model's tokenizer systematically yields more provider
        tokens than this heuristic counts — that makes the engine protect a LARGER real
        tail; lower it only when over-protection is measurably wasteful.
        """
        sid = self._session_id or ""
        if conn is None or self._store is None:
            return 1.0
        frozen = self._store.get_meta_in_tx(conn, sid, "estimate_ratio", None)
        if frozen is not None:
            try:
                return _clamp(float(frozen), MIN_RATIO, MAX_RATIO)
            except (TypeError, ValueError):
                return 1.0
        ratio = _clamp(_env_float("HCE_ESTIMATE_RATIO", 1.0), MIN_RATIO, MAX_RATIO)
        self._store.set_meta(conn, sid, "estimate_ratio", round(ratio, 6))
        return ratio

    def _protected_tag_numbers(
        self, conn: Any, scan: Any, unit_tag_numbers: Optional[dict[tuple, int]] = None,
    ) -> set[int]:
        """Token-space tail window + protected_tools. Only persistent data participates
        (design 3.1 顺序无关性)."""
        protected: set[int] = set()
        unit_tag_numbers = unit_tag_numbers or {}
        usable = int(self.context_length or 0) * (self._threshold_pct / 100.0)
        tool_units = [u for u in scan.units if u.kind == "tool"]
        if usable <= 0:
            span_start = len(scan.identities)  # no window math: structural protection only
        else:
            ratio = self._estimate_ratio(conn)
            floor = min(PROTECTION_FLOOR_ABS, round(PROTECTION_FLOOR_FRAC * usable))
            target = _clamp(round(PROTECTION_FRACTION * usable), floor, PROTECTION_CEILING)
            target_units = target / ratio
            acc = 0
            span_start = 0
            for i in range(len(scan.token_estimates) - 1, -1, -1):
                acc += scan.token_estimates[i]
                if acc >= target_units:
                    span_start = i
                    break
        for unit in scan.units:
            number = unit_tag_numbers.get(
                (unit.kind, unit.carrier_id, unit.call_id, unit.part_index)
            )
            if number is None:
                continue
            # Protection is per ARC, not per part: protecting a result while its owner
            # assistant sits outside the window would strip the call and orphan the
            # result (a shape some providers reject outright).
            inside = unit.carrier_index >= span_start
            if not inside and unit.owner_index is not None:
                inside = unit.owner_index >= span_start
            if inside:
                protected.add(number)
        # Structural minimum: the newest few tool arcs stay visible no matter where the
        # token window lands, so the model never loses sight of what a tool call looks
        # like (it starts inventing them otherwise).
        for unit in tool_units[-STRUCTURAL_TOOL_MINIMUM:]:
            number = unit_tag_numbers.get(
                (unit.kind, unit.carrier_id, unit.call_id, unit.part_index)
            )
            if number is not None:
                protected.add(number)
        reduce_calls = [u for u in tool_units if u.call_name in PROTECTED_TOOL_KEEP]
        for unit in reduce_calls[-PROTECTED_TOOL_KEEP["ctx_reduce"]:]:
            number = unit_tag_numbers.get(
                (unit.kind, unit.carrier_id, unit.call_id, unit.part_index)
            )
            if number is not None:
                protected.add(number)
        return protected

    @staticmethod
    def _tag_token(tag_rows: dict[int, dict[str, Any]], number: int) -> int:
        row = tag_rows.get(number)
        return int(row["token_count"]) if row and row.get("token_count") is not None else 0

    # ------------------------------------------------------------------
    # LKG
    # ------------------------------------------------------------------

    def _update_lkg(self, conn: Any, output: list[Any], head_len: int, scan: Any) -> None:
        import hashlib

        sid = self._session_id or ""
        blob = json.dumps(output, ensure_ascii=False, default=str)
        out_hash = hashlib.sha1(blob.encode("utf-8")).hexdigest()
        existing = conn.execute("SELECT out_hash FROM lkg WHERE session_id=?", (sid,)).fetchone()
        if existing is not None and existing["out_hash"] == out_hash:
            return
        self._store.write_lkg(conn, {
            "session_id": sid,
            "payload": json.dumps(output[head_len:], ensure_ascii=False, default=str),
            "out_hash": out_hash,
            "head_len": int(head_len),
            "n_history": int(len(output) - head_len),
            "head_fps": json.dumps(
                [content_fingerprint(m) for m in output[:head_len] if isinstance(m, dict)]
            ),
            "base_fps": json.dumps(list(scan.fingerprints)),
            "model": str(getattr(self, "model", "") or ""),
            "updated_at": time.time(),
        })

    def _lkg_replay(self, request_messages: list[dict[str, Any]]) -> Optional[list[dict[str, Any]]]:
        """Fail-frozen degradation: verify anchors, then replay the last served bytes
        plus the raw new tail. Without a verifiable LKG, return None (host sends
        unmodified) and log loudly — never manage half-blind."""
        try:
            if not self._session_id or self._store is None:
                return None
            record = self._store.load_lkg(self._session_id)
            if not record:
                return None
            # A snapshot rendered under a different model is not replayable: pricing and
            # provider shapes differ. Mismatch ⇒ drop the slot rather than serve stale bytes.
            snapshot_model = str(record.get("model") or "")
            current_model = str(self.model or "")
            if snapshot_model and current_model and snapshot_model != current_model:
                logger.info(
                    "HCE LKG dropped: model changed %s -> %s", snapshot_model, current_model
                )
                with self._store.transaction() as conn:
                    conn.execute("DELETE FROM lkg WHERE session_id=?", (self._session_id,))
                return None
            payload = json.loads(record["payload"])
            head_fps = json.loads(record["head_fps"])
            base_fps = json.loads(record["base_fps"])
            head_len = int(record["head_len"])
            n_history = int(record["n_history"])
            if len(request_messages) < head_len:
                return None
            for j, fp in enumerate(head_fps):
                msg = request_messages[j]
                if not isinstance(msg, dict) or content_fingerprint(msg) != fp:
                    return None
            if len(request_messages) - head_len < n_history:
                return None
            for i in range(n_history):
                msg = request_messages[head_len + i]
                if not isinstance(msg, dict) or content_fingerprint(msg) != base_fps[i]:
                    return None
            head = [dict(m) for m in request_messages[:head_len] if isinstance(m, dict)]
            replayed = [dict(m) for m in payload if isinstance(m, dict)]
            tail = [dict(m) for m in request_messages[head_len + n_history:] if isinstance(m, dict)]
            logger.warning("HCE replaying LKG after internal failure (session=%s)", self._session_id)
            return head + replayed + tail
        except Exception:
            logger.error("HCE LKG replay unavailable; sending unmodified request", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # compress()
    # ------------------------------------------------------------------

    def compress(
        self, messages: list[dict[str, Any]], current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None, force: bool = False, memory_context: str = "",
        bypass_cooldown: bool = False,
    ) -> list[dict[str, Any]]:
        """Emergency lane: land every held drop, skeletonize oldest tool outputs until
        the history fits ~60% of the window, return the list (host commits it as the
        new transcript). Never raises; returns ``messages`` unchanged on any failure."""
        try:
            if not self._enabled or not self._session_id or not isinstance(messages, list):
                return messages
            store = self._open_store()
            if store is None:
                return messages
            now = time.time()
            with store.transaction() as conn:
                sid = self._session_id
                scan = scan_conversation(messages)
                unit_tag_numbers, tag_rows, _changed = self._reconcile(
                    conn, scan, messages, now, allow_compaction=False
                )
                landed = self._land_pending(
                    conn, messages, scan, unit_tag_numbers, tag_rows,
                    store.load_pending(conn, sid), None, now,
                )
                target = self._compress_target_units(conn, scan)
                skeletonized = self._skeletonize_oldest(
                    conn, messages, scan, unit_tag_numbers, tag_rows, target
                )
                output, _markers, _head = render_request(
                    messages, messages, scan, unit_tag_numbers, tag_rows, {},
                    include_markers=False, prefix_active=False, skip_verification=True,
                )
                if output != messages or landed or skeletonized:
                    self.compression_count = int(getattr(self, "compression_count", 0)) + 1
                    self._bump(conn, "compressions")
                    logger.info(
                        "HCE compress: %d messages -> %d (session=%s)",
                        len(messages), len(output), sid,
                    )
                return output
        except Exception:
            logger.warning("HCE compress failed; returning messages unchanged", exc_info=True)
            return messages

    def _compress_target_units(self, conn: Any, scan: Any) -> Optional[int]:
        ctx_len = int(self.context_length or 0)
        if ctx_len <= 0:
            return sum(scan.token_estimates) // 2  # no window info: shed half
        ratio = self._estimate_ratio(conn)
        return int(0.60 * ctx_len / max(ratio, MIN_RATIO))

    def _skeletonize_oldest(
        self, conn: Any, conversation: list[dict[str, Any]], scan: Any,
        unit_tag_numbers: dict[tuple, int], tag_rows: dict[int, dict[str, Any]],
        target_units: Optional[int],
    ) -> bool:
        sid = self._session_id or ""
        store = self._store
        assert store is not None
        if target_units is None:
            return False
        placeholder_cost = len(PLACEHOLDER.format(n=0)) / 4
        # Start from the POST-LANDING estimate: tags _land_pending just dropped no
        # longer carry their full tokens (placeholder cost only).
        active_tokens = 0
        dropped_units = 0
        for unit in scan.units:
            number = unit_tag_numbers.get(
                (unit.kind, unit.carrier_id, unit.call_id, unit.part_index)
            )
            row = tag_rows.get(number) if number is not None else None
            if row is not None and row.get("status") == "dropped":
                dropped_units += 1
            else:
                active_tokens += int(unit.token_count)
        current = active_tokens + dropped_units * placeholder_cost
        changed = False
        for unit in scan.units:  # scan order == conversation order == oldest first
            if unit.kind != "tool":
                continue
            if current <= target_units:
                break
            if unit.result_is_last:
                continue  # never blind the live tail (design 9.8)
            number = unit_tag_numbers.get(
                (unit.kind, unit.carrier_id, unit.call_id, unit.part_index)
            )
            if number is None:
                continue
            row = tag_rows.get(number)
            if row is None or row.get("status") != "active":
                continue
            payload = self._source_payload(conversation, unit)
            if payload is not None:
                store.store_source(conn, sid, number, payload)
            store.update_tag_status(conn, sid, number, "dropped", drop_form="skeleton")
            store.dequeue(conn, sid, number)
            tag_rows[number] = {**row, "status": "dropped", "drop_form": "skeleton"}
            current -= max(int(unit.token_count) - placeholder_cost, 0)
            changed = True
        return changed

    # ------------------------------------------------------------------
    # tools
    # ------------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return tool_schemas() if self._enabled else []

    def handle_tool_call(self, name: str, args: dict[str, Any], **kwargs: Any) -> str:
        try:
            args = args if isinstance(args, dict) else {}
            if name == "ctx_reduce":
                return self._tool_reduce(args, kwargs.get("messages"))
            if name == "ctx_expand":
                return self._tool_expand(args)
            return json.dumps({"error": f"Unknown context engine tool: {name}"})
        except Exception as exc:
            logger.warning("HCE tool %s failed", name, exc_info=True)
            return json.dumps({"error": f"ctx tool failed: {exc}"})

    def _tool_reduce(self, args: dict[str, Any], messages: Any) -> str:
        ids, bad = parse_tag_ids(args.get("tags"))
        if bad:
            logger.debug("HCE ctx_reduce unparsed tokens: %r", bad)
        if not ids:
            return json.dumps({
                "error": 'No tag ids parsed. Pass tags like "3-5" or "1,2,9" — the §N§ '
                         "markers shown on tool results and file parts."
            })
        store = self._open_store()
        if store is None or not self._session_id:
            return json.dumps(
                {"error": "Context engine is not bound to a session yet; retry shortly."}
            )
        now = time.time()
        with store.transaction() as conn:
            sid = self._session_id
            rows = {int(r["tag_number"]): dict(r) for r in store.load_tags_all(conn, sid)}
            scan = scan_conversation(messages) if isinstance(messages, list) else None
            unit_map: dict[int, Unit] = {}
            protected: set[int] = set()
            if scan is not None:
                unit_map, protected = self._match_existing(conn, scan, rows)
            stamped: list[int] = []
            held: list[int] = []
            already: list[int] = []
            skipped_self: list[int] = []
            refused: list[int] = []
            unknown: list[int] = []
            for number in ids:
                row = rows.get(number)
                if row is None:
                    unknown.append(number)
                elif row["status"] == "compacted":
                    refused.append(number)
                elif row["status"] == "dropped" or store.pending_exists(conn, sid, number):
                    already.append(number)
                elif row["kind"] == "tool" and row["call_name"] in PROTECTED_TOOL_KEEP:
                    skipped_self.append(number)
                else:
                    form = self._decide_form(unit_map.get(number), scan)
                    store.queue_drop(conn, sid, number, form, now)
                    (held if number in protected else stamped).append(number)
            max_tag = store.next_tag_number(conn, sid) - 1
            self._bump(conn, "reduce_calls")
            self._bump(conn, "reduce_stamped", len(stamped))
            self._bump(conn, "reduce_held", len(held))
            self._bump(conn, "reduce_already", len(already))
            self._bump(conn, "reduce_refused", len(refused) + len(unknown))
        return reduce_response(stamped, held, already, skipped_self, refused, unknown, max_tag)

    def _match_existing(
        self, conn: Any, scan: Any, rows: dict[int, dict[str, Any]],
    ) -> tuple[dict[int, Unit], set[int]]:
        """Lightweight match (no inserts) for tool handlers: unit per tag + protection."""
        unit_map: dict[int, Unit] = {}
        unit_tag_numbers: dict[tuple, int] = {}
        used: set[int] = set()
        for unit in scan.units:
            key = (unit.kind, unit.carrier_id, unit.call_id, unit.part_index)
            for number, row in rows.items():
                if number in used or row["kind"] != unit.kind:
                    continue
                if row["call_id"] not in (None, unit.call_id):
                    continue
                if row["part_index"] not in (None, unit.part_index):
                    continue
                if row["uid"] and row["uid"] != unit.uid:
                    continue
                if row["content_hash"] == unit.content_hash or (row["uid"] and row["uid"] == unit.uid):
                    unit_map[number] = unit
                    unit_tag_numbers[key] = number
                    used.add(number)
                    break
        protected = self._protected_tag_numbers(conn, scan, unit_tag_numbers)
        return unit_map, protected

    def _decide_form(self, unit: Optional[Unit], scan: Any) -> str:
        if unit is None or unit.kind == "file":
            return "placeholder"
        small = (unit.args_bytes or 0) <= SMALL_ARGS_BYTES
        recent = False
        if scan is not None:
            arcs = [u for u in scan.units if u.kind == "tool"]
            recent_ids = {id(u) for u in arcs[-RECENT_CALL_WINDOW:]}
            recent = id(unit) in recent_ids
        return "skeleton" if (recent and small) else "remove"

    def _tool_expand(self, args: dict[str, Any]) -> str:
        ids, _bad = parse_tag_ids(args.get("tags"))
        if not ids:
            return json.dumps({"error": 'No tag ids parsed. Pass tags like "7" or "3-5".'})
        store = self._open_store()
        if store is None or not self._session_id:
            return json.dumps(
                {"error": "Context engine is not bound to a session yet; retry shortly."}
            )
        sid = self._session_id
        sections: list[tuple[int, str]] = []
        notes: list[str] = []
        max_tag = store.max_tag_number(sid)
        for number in ids:
            source = store.load_source(sid, number)
            if source is not None:
                sections.append((number, self._format_source(source)))
                continue
            status = store.tag_status(sid, number)
            if status is None:
                if max_tag == 0:
                    notes.append(
                        f"Tag §{number}§ does not exist — no tags have been created in "
                        "this session yet."
                    )
                else:
                    notes.append(
                        f"Tag §{number}§ does not exist. Valid tags run §1§–§{max_tag}§."
                    )
            elif status == "active":
                notes.append(
                    f"Tag §{number}§ is still live in the context; nothing was dropped yet."
                )
            else:
                notes.append(
                    f"Tag §{number}§ left the transcript before being stamped; "
                    "no original is stored."
                )
        max_tag = store.max_tag_number(sid)
        self._bump_safe("expand_calls")
        if sections:
            self._bump_safe("expand_hits", len(sections))
        return expand_response(sections, notes)

    def _format_source(self, source: dict[str, Any]) -> str:
        kind = source.get("kind")
        if kind == "tool":
            chunks = []
            if source.get("call_name"):
                chunks.append(f"tool call: {source['call_name']}")
            if source.get("arguments") is not None:
                args_value = source["arguments"]
                if not isinstance(args_value, str):
                    args_value = json.dumps(args_value, ensure_ascii=False)
                chunks.append(f"arguments: {args_value}")
            chunks.append("result:")
            chunks.append(str(source.get("result") or ""))
            return "\n".join(chunks)
        if kind == "file":
            part = source.get("part")
            if isinstance(part, dict):
                return json.dumps(part, ensure_ascii=False)
            return str(part or "")
        return json.dumps(source, ensure_ascii=False)

    # ------------------------------------------------------------------
    # plumbing
    # ------------------------------------------------------------------

    def _open_store(self) -> Optional[Store]:
        if self._store is not None:
            return self._store
        if self._store_failed:
            return None
        path = self._resolve_store_path()
        if path is None:
            self._warn_once("no hermes home resolved; context engine idle")
            self._store_failed = True
            return None
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._store = Store(str(path))
            return self._store
        except SchemaFenceError as exc:
            self._store_failed = True
            logger.error("HCE schema fence: %s; returning unmodified requests from now on", exc)
            return None
        except Exception:
            logger.warning("HCE store open failed at %s", path, exc_info=True)
            self._store_failed = True
            return None

    def _resolve_store_path(self) -> Optional[str]:
        home = self._hermes_home or os.environ.get("HERMES_HOME")
        if not home:
            try:
                from hermes_constants import get_hermes_home

                home = str(get_hermes_home())
            except Exception:
                home = None
        if not home:
            return None
        return str(Path(home) / "context-engine" / "context.db")

    def _warn_once(self, message: str) -> None:
        if not self._warned:
            self._warned = True
            logger.warning("HCE: %s", message)

    def clone_for_agent(self) -> "HermesContextEngine":
        """All durable state lives in the shared store; per-agent instances carry only
        config + token counters (共享持久后端，仅拷贝可变预算状态)."""
        clone = HermesContextEngine()
        clone._hermes_home = self._hermes_home
        clone.threshold_percent = self.threshold_percent
        clone._threshold_pct = self._threshold_pct
        clone.context_length = self.context_length
        clone.threshold_tokens = self.threshold_tokens
        return clone
