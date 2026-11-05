"""Unit identification, estimation, markers and the schema fence."""

from __future__ import annotations

import json
import sqlite3

from hce_plugin.estimate import (
    args_string_bytes,
    estimate_message_tokens,
    estimate_text_tokens,
)
from hce_plugin.render import marker_text_for
from hce_plugin.units import scan_conversation


def _assistant(uid, calls, text=""):
    return {
        "role": "assistant", "content": text, "tool_calls": calls,
        "message_uid": uid, "timestamp": 1_700_000_000.0,
    }


def _tool(uid, call_id, content="r"):
    return {
        "role": "tool", "tool_call_id": call_id, "content": content,
        "message_uid": uid, "timestamp": 1_700_000_060.0,
    }


def test_pairing_prefers_earliest_unpaired_call():
    call = lambda cid: {"id": cid, "type": "function",
                        "function": {"name": "f", "arguments": "{}"}}
    conversation = [
        _assistant("a1", [call("c1"), call("c2")]),
        _tool("t1", "c2"),
        _tool("t2", "c1"),
        _assistant("a2", [call("c1")]),
        _tool("t3", "c1"),
    ]
    scan = scan_conversation(conversation)
    tool_units = [u for u in scan.units if u.kind == "tool"]
    assert len(tool_units) == 3
    assert tool_units[0].owner_id == "u:a1" and tool_units[0].call_id == "c2"
    assert tool_units[1].owner_id == "u:a1" and tool_units[1].call_id == "c1"
    assert tool_units[2].owner_id == "u:a2" and tool_units[2].call_id == "c1"


def test_pairing_prefers_occurrence_uid_over_repeated_provider_id():
    """真机形态：provider call id 跨轮重复，靠 _tool_call_uid 精确配对。"""
    call = lambda cid: {"id": cid, "type": "function",
                        "function": {"name": "f", "arguments": "{}"}}
    a1 = _assistant("a1", [call("c1")])
    a1["_tool_call_uids"] = {"c1": "occ-1"}
    a2 = _assistant("a2", [call("c1")])
    a2["_tool_call_uids"] = {"c1": "occ-2"}
    t1 = _tool("t1", "c1")
    t1["_tool_call_uid"] = "occ-2"
    t2 = _tool("t2", "c1")
    t2["_tool_call_uid"] = "occ-1"
    conversation = [a1, a2, t1, t2]
    scan = scan_conversation(conversation)
    tool_units = [u for u in scan.units if u.kind == "tool"]
    assert len(tool_units) == 2
    by_carrier = {u.carrier_id: u for u in tool_units}
    assert by_carrier["u:t1"].owner_id == "u:a2"   # occ-2 pairs with a2 despite order
    assert by_carrier["u:t2"].owner_id == "u:a1"


def test_unpaired_result_gets_no_owner():
    conversation = [_tool("t1", "c9", "orphan")]
    scan = scan_conversation(conversation)
    assert scan.units[0].owner_id is None
    assert scan.units[0].kind == "tool"


def test_file_parts_are_units():
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "text", "text": "and this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}},
            ],
            "message_uid": "u-img", "timestamp": 1_700_000_000.0,
        }
    ]
    scan = scan_conversation(conversation)
    files = [u for u in scan.units if u.kind == "file"]
    assert [u.part_index for u in files] == [1, 3]


def test_estimator_cjk_and_ascii():
    assert estimate_text_tokens("hello world") == 3       # ceil(11/3.8)
    assert estimate_text_tokens("你好世界") == 4           # CJK ~1 token/char
    assert estimate_text_tokens("") == 0


def test_args_bytes_counts_values_only():
    payload = {"path": "x" * 10, "count": 5, "flags": [True, None], "nested": {"k": "y" * 4}}
    assert args_string_bytes(payload) == 14
    assert args_string_bytes('{"path": "' + "z" * 6 + '"}') == 6


def test_message_estimate_prices_parts_and_calls():
    msg = {
        "role": "user",
        "content": [{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {}}],
    }
    assert estimate_message_tokens(msg) == 2 + 1500  # ceil(5/3.8) + flat image price


def test_marker_text_format():
    assert marker_text_for(90.0) == ""                    # < 5min: no marker at all
    assert marker_text_for(299.0) == ""
    assert marker_text_for(300.0) == "<!-- +5m -->"
    assert marker_text_for(90 * 60.0) == "<!-- +1h 30m -->"
    assert marker_text_for(2 * 3600.0) == "<!-- +2h -->"
    assert marker_text_for(3 * 86400.0) == "<!-- +3d -->"
    assert marker_text_for(86400.0 + 7200.0) == "<!-- +1d 2h -->"
    assert marker_text_for(14 * 86400.0) == "<!-- +2w -->"
    assert marker_text_for(None) == ""
    assert marker_text_for(-3.0) == ""


def test_schema_fence_refuses_newer_database(make_engine, tmp_path):
    db_dir = tmp_path / "context-engine"
    db_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_dir / "context.db")
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta (key, value) VALUES ('schema_version', '999')")
    conn.commit()
    conn.close()

    engine = make_engine()
    conversation = [_tool("t1", "c1")]
    request = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    assert engine.select_context(
        request, conversation_messages=conversation, budget_tokens=0
    ) is None
    assert engine.compress(conversation) == conversation
    assert engine._store_failed is True


def test_estimate_ratio_frozen_default_and_override(make_engine, monkeypatch):
    """比值每会话冻结一次：默认 1.0，可用 HCE_ESTIMATE_RATIO 显式给，之后不再变。"""
    from host_sim import build_request, make_turn

    engine = make_engine(context_length=1_000_000)
    conversation: list[dict] = []
    for i in range(3):
        conversation.extend(make_turn(i))
    output = engine.select_context(
        build_request(conversation), conversation_messages=conversation, budget_tokens=0
    )
    assert output is not None
    assert engine._store.get_meta(engine._session_id, "estimate_ratio") == 1.0

    # Provider usage must NOT rewrite the ratio (no runtime learning).
    engine.update_from_response({"prompt_tokens": 16809})
    assert engine._store.get_meta(engine._session_id, "estimate_ratio") == 1.0
    assert engine.last_prompt_tokens == 16809

    # An explicit env override applies to a FRESH session and then stays frozen,
    # even after the env changes under a live session.
    monkeypatch.setenv("HCE_ESTIMATE_RATIO", "1.6")
    fresh = make_engine(session_id="s2", context_length=1_000_000)
    fresh.select_context(
        build_request(conversation), conversation_messages=conversation, budget_tokens=0
    )
    assert fresh._store.get_meta("s2", "estimate_ratio") == 1.6
    monkeypatch.setenv("HCE_ESTIMATE_RATIO", "0.5")
    assert fresh._estimate_ratio(None) == 1.0  # conn None -> neutral, never re-reads
    with fresh._store.transaction() as conn:
        assert fresh._estimate_ratio(conn) == 1.6
