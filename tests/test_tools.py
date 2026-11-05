"""Tool-face contracts: tag id parsing, the four reduce semantics, expand retrieval."""

from __future__ import annotations

import json
import sqlite3

from hce_plugin.tools import parse_tag_ids
from host_sim import build_request, make_turn


def build_conversation(counter=None):
    """Three small arcs + one large tail arc: the tail keeps tags 1-3 outside the
    protection window so ctx_reduce stamps (rather than holds) them."""
    from host_sim import make_turn as _mt

    rows: list[dict] = []
    for i in range(3):
        rows.extend(_mt(i))
    rows.extend(
        _mt(90, tool_result="x" * 40_000,
            tool_args=json.dumps({"path": "/data/file_90.bin"}))
    )
    return rows


def seed_engine(make_engine, conversation):
    engine = make_engine()
    engine.update_from_response({"prompt_tokens": 8_000})
    request = build_request(conversation)
    output = engine.select_context(
        request, conversation_messages=conversation, budget_tokens=engine.context_length
    )
    assert output is not None
    return engine, output


def test_parse_tag_ids_forms():
    assert parse_tag_ids("3-5") == ([3, 4, 5], [])
    assert parse_tag_ids("1,2,9") == ([1, 2, 9], [])
    assert parse_tag_ids("§7§") == ([7], [])
    assert parse_tag_ids("3-3") == ([3], [])
    assert parse_tag_ids("9-3")[1] == ["9-3"]
    assert parse_tag_ids("0")[1] == ["0"]
    assert parse_tag_ids(["1", "2-2"]) == ([1, 2], [])


def test_reduce_stamps_and_expand_returns_original(make_engine, tmp_path):
    conversation = build_conversation()
    engine, output = seed_engine(make_engine, conversation)
    response = json.loads(engine.handle_tool_call("ctx_reduce", {"tags": "1"}, messages=conversation))
    assert response["stamped"] == [1]
    assert "queued, not deleted" in response["summary"]

    # Land it via a queue-triggering pass? Small content stays held/deferred —
    # force the landing through compress (the emergency lane force-lands).
    compressed = engine.compress(conversation)
    texts = [m.get("content") for m in compressed]
    assert "[dropped §1§]" in texts

    expanded = json.loads(engine.handle_tool_call("ctx_expand", {"tags": "1"}))
    assert "data" in expanded["content"]
    assert "=== §1§ ===" in expanded["content"]


def test_reduce_four_semantics(make_engine, tmp_path):
    conversation = build_conversation()
    engine, _output = seed_engine(make_engine, conversation)

    first = json.loads(engine.handle_tool_call("ctx_reduce", {"tags": "1"}, messages=conversation))
    assert first["stamped"] == [1]
    again = json.loads(engine.handle_tool_call("ctx_reduce", {"tags": "1"}, messages=conversation))
    assert again["already"] == [1]
    assert "No new action is needed" in again["summary"]

    unknown = json.loads(engine.handle_tool_call("ctx_reduce", {"tags": "99"}, messages=conversation))
    assert unknown["unknown"] == [99]
    assert "do not exist" in unknown["summary"]

    # A live ctx_reduce arc must be refused with the "cleans up automatically"
    # copy instead of being stamped.
    from host_sim import make_turn as _mt
    with_self = conversation + _mt(
        99, tool_name="ctx_reduce", tool_args='{"tags": "1-3"}',
        tool_result="stamped 3 tags",
    )
    engine.update_from_response({"prompt_tokens": 9_000})
    out2 = engine.select_context(
        build_request(with_self), conversation_messages=with_self,
        budget_tokens=engine.context_length,
    )
    assert out2 is not None
    self_call_id = next(
        call["id"] for m in with_self if m.get("role") == "assistant"
        for call in m.get("tool_calls", [])
        if call.get("function", {}).get("name") == "ctx_reduce"
    )
    self_result = next(m for m in out2 if m.get("role") == "tool"
                       and m.get("tool_call_id") == self_call_id)
    self_tag = int(self_result["content"].split(" ", 1)[0][1:-1])
    self_ref2 = json.loads(
        engine.handle_tool_call("ctx_reduce", {"tags": str(self_tag)}, messages=with_self)
    )
    assert self_ref2["skipped_self"] == [self_tag]

    # compacted refusal: force the status directly in the store
    store = engine._store
    with store.transaction() as conn:
        store.update_tag_status(conn, engine._session_id, 2, "compacted")
    # tag 2 was never queued, so queue it first is impossible post-compaction —
    # stamping it now must be refused.
    refused = json.loads(engine.handle_tool_call("ctx_reduce", {"tags": "2"}, messages=conversation))
    assert refused["refused"] == [2]
    assert "compression boundary" in refused["summary"]


def test_expand_semantics(make_engine):
    conversation = build_conversation()
    engine, _output = seed_engine(make_engine, conversation)

    active = json.loads(engine.handle_tool_call("ctx_expand", {"tags": "1"}))
    assert active["sections"] == [] or active["content"] == ""
    assert "still live in the context" in active["summary"]

    missing = json.loads(engine.handle_tool_call("ctx_expand", {"tags": "77"}))
    assert "does not exist" in missing["summary"]
    assert "§1§" in missing["summary"]  # valid range hint

    bad_input = json.loads(engine.handle_tool_call("ctx_expand", {"tags": ""}))
    assert "error" in bad_input


def test_reduce_without_messages_defaults_safe(make_engine):
    conversation = build_conversation()
    engine, _output = seed_engine(make_engine, conversation)
    # No live list available: form decision must stay non-destructive.
    response = json.loads(engine.handle_tool_call("ctx_reduce", {"tags": "1"}))
    assert response["stamped"] == [1]
    store = engine._store
    with store.transaction() as conn:
        rows = store.load_pending(conn, engine._session_id)
    forms = {r["tag_number"]: r["drop_form"] for r in rows}
    assert forms[1] == "placeholder"
