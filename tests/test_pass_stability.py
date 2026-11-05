"""Pass-level behaviour: byte stability (AC1), landing + restability (AC2),
restart reproduction (AC3), markers, prefixes and fail-open mapping."""

from __future__ import annotations

import json

from hce_plugin.render import PLACEHOLDER
from host_sim import build_request, fingerprint_list, make_turn


def run_pass(engine, conversation):
    request = build_request(conversation)
    output = engine.select_context(
        request, conversation_messages=conversation, budget_tokens=engine.context_length
    )
    assert output is not None
    # The host's request list must stay pristine (copy-based render).
    assert output[0]["content"] == request[0]["content"]
    return output


class Counter:
    def __init__(self) -> None:
        self.n = 0

    def turn(self, **kwargs) -> list[dict]:
        rows = make_turn(self.n, **kwargs)
        self.n += 1
        return rows

    def big(self, size=40_000, **kwargs) -> list[dict]:
        return self.turn(tool_result="x" * size,
                         tool_args=json.dumps({"path": f"/data/file_{self.n}.bin"}),
                         **kwargs)


def find_tag_for(output, needle: str) -> int:
    """The §N§ prefix of the tool result containing ``needle``."""
    for msg in output:
        if msg.get("role") == "tool" and isinstance(msg.get("content"), str) \
                and needle in msg["content"]:
            head = msg["content"].split(" ", 1)[0]
            assert head.startswith("§") and head.endswith("§"), msg["content"][:20]
            return int(head[1:-1])
    raise AssertionError(f"no tool result containing {needle!r}")


def test_ac1_fifty_turns_append_only(make_engine):
    engine = make_engine()
    counter = Counter()
    conversation: list[dict] = []
    previous: list | None = None
    for _ in range(50):
        conversation.extend(counter.turn())
        engine.update_from_response({"prompt_tokens": 5_000 + counter.n * 10})
        output = run_pass(engine, conversation)
        if previous is not None:
            # Everything previously served must reappear byte-identical.
            assert fingerprint_list(output[: len(previous)]) == fingerprint_list(previous)
        previous = output


def test_ac2_reduce_lands_once_then_stays_stable(make_engine):
    engine = make_engine()
    counter = Counter()
    conversation: list[dict] = []
    for _ in range(3):
        conversation.extend(counter.turn())
    conversation.extend(counter.big())          # tag A: the drop target
    for _ in range(2):
        conversation.extend(counter.turn())
    conversation.extend(counter.big())          # newer big content keeps A outside the window
    conversation.extend(counter.turn())
    engine.update_from_response({"prompt_tokens": 30_000})
    before = run_pass(engine, conversation)
    target_tag = find_tag_for(before, "x" * 1000)
    assert target_tag == 4  # turns 0-2 own tags 1-3

    response = engine.handle_tool_call(
        "ctx_reduce", {"tags": str(target_tag)}, messages=conversation
    )
    payload = json.loads(response)
    assert payload["stamped"] == [target_tag]

    # Next pass: the queue gate fires, the drop lands.
    engine.update_from_response({"prompt_tokens": 31_000})
    landed = run_pass(engine, conversation)
    landed_texts = [m.get("content") for m in landed]
    assert PLACEHOLDER.format(n=target_tag) in landed_texts

    # The pass after that must be byte-identical to the landing pass.
    engine.update_from_response({"prompt_tokens": 31_500})
    after = run_pass(engine, conversation)
    assert fingerprint_list(after) == fingerprint_list(landed)
    assert fingerprint_list(after) != fingerprint_list(before)


def test_ac3_restart_reproduces_bytes(make_engine):
    engine = make_engine()
    counter = Counter()
    conversation: list[dict] = []
    for _ in range(3):
        conversation.extend(counter.turn())
    conversation.extend(counter.big())
    for _ in range(2):
        conversation.extend(counter.turn())
    conversation.extend(counter.big())
    for _ in range(2):
        conversation.extend(counter.turn())
    engine.update_from_response({"prompt_tokens": 30_000})
    run_pass(engine, conversation)
    target_tag = None
    # stamp via the handler using the conversation as the live list
    probe = run_pass(engine, conversation)
    target_tag = find_tag_for(probe, "x" * 1000)
    response = engine.handle_tool_call(
        "ctx_reduce", {"tags": str(target_tag)}, messages=conversation
    )
    assert json.loads(response)["stamped"] == [target_tag]
    engine.update_from_response({"prompt_tokens": 31_000})
    landed = run_pass(engine, conversation)

    # Kill + restart: a brand-new engine instance (fresh store connection, empty
    # memory) must reproduce the exact same bytes on the same inputs.
    reborn = make_engine()
    reborn.update_from_response({"prompt_tokens": 31_000})
    replayed = run_pass(reborn, conversation)
    assert fingerprint_list(replayed) == fingerprint_list(landed)


def test_remove_form_for_large_args_arc(make_engine):
    """大参数弧 → 整体移除：结果消息消失、owner 的 tool_calls 摘除该调用。"""
    engine = make_engine()
    counter = Counter()
    conversation: list[dict] = []
    for _ in range(2):
        conversation.extend(counter.turn())
    big_args = json.dumps({"path": "/data/", "blob": "a" * 2000})
    conversation.extend(
        counter.turn(tool_args=big_args, tool_result="tiny result")
    )
    for _ in range(2):
        conversation.extend(counter.turn())
    conversation.extend(counter.big())  # tail keeps the target outside the window
    engine.update_from_response({"prompt_tokens": 30_000})
    out = run_pass(engine, conversation)
    target_call_id = next(
        call["id"] for m in conversation if m.get("role") == "assistant"
        for call in m.get("tool_calls", [])
        if "blob" in call.get("function", {}).get("arguments", "")
    )
    target_result = next(m for m in out if m.get("role") == "tool"
                         and m.get("tool_call_id") == target_call_id)
    target_tag = int(target_result["content"].split(" ", 1)[0][1:-1])

    response = engine.handle_tool_call(
        "ctx_reduce", {"tags": str(target_tag)}, messages=conversation
    )
    payload = json.loads(response)
    assert payload["stamped"] == [target_tag]

    # Threshold crossing permits the rebuild; the removal lands.
    engine.update_from_response({"prompt_tokens": 140_000})
    landed = run_pass(engine, conversation)
    assert not any(
        m.get("role") == "tool" and m.get("tool_call_id") == target_call_id
        for m in landed
    )
    owner_index = next(
        i for i, m in enumerate(conversation) if m.get("role") == "assistant"
        and any(c.get("id") == target_call_id for c in m.get("tool_calls", []))
    )
    head_len = len(landed) - len(conversation)          # system/prefill rows
    owner = landed[owner_index + head_len]
    assert not owner.get("tool_calls"), "big-args call must be stripped from its owner"
    assert owner["content"], "owner keeps its text and therefore its row"

    # Subsequent passes stay byte-identical (removal re-derived from state).
    engine.update_from_response({"prompt_tokens": 141_000})
    again = run_pass(engine, conversation)
    assert fingerprint_list(again) == fingerprint_list(landed)


def test_tag_prefix_visible_on_active_results(make_engine):
    engine = make_engine()
    counter = Counter()
    conversation: list[dict] = []
    for _ in range(2):
        conversation.extend(counter.turn())
    output = run_pass(engine, conversation)
    tool_texts = [m["content"] for m in output if m.get("role") == "tool"]
    assert tool_texts[0].startswith("§1§ ")
    assert tool_texts[1].startswith("§2§ ")


def test_time_markers_frozen_and_replayed(make_engine):
    engine = make_engine()
    conversation: list[dict] = []
    # 10-minute gaps: above the 5-minute marker threshold.
    conversation.extend(make_turn(0, start_ts=1_700_000_000.0))
    conversation.extend(make_turn(1, start_ts=1_700_000_600.0))
    first = run_pass(engine, conversation)
    user_texts = [m["content"] for m in first if m.get("role") == "user"]
    # First message has no predecessor -> no marker; second sees a ~9.5min gap.
    assert user_texts[0] == "task 0"
    assert user_texts[1].startswith("<!-- +9m -->\n")
    conversation.extend(make_turn(2, start_ts=1_700_001_200.0))
    second = run_pass(engine, conversation)
    # Frozen markers replay byte-exact on later passes.
    assert [m["content"] for m in second if m.get("role") == "user"][:2] == user_texts


def test_short_gaps_get_no_marker(make_engine):
    engine = make_engine()
    conversation: list[dict] = []
    conversation.extend(make_turn(0))
    conversation.extend(make_turn(1))  # host_sim default gap is 120s < 5min
    out = run_pass(engine, conversation)
    assert [m["content"] for m in out if m.get("role") == "user"] == ["task 0", "task 1"]


def test_held_drop_stays_queued_until_exposed(make_engine):
    engine = make_engine()
    counter = Counter()
    conversation: list[dict] = []
    for _ in range(2):
        conversation.extend(counter.turn())
    engine.update_from_response({"prompt_tokens": 8_000})
    run_pass(engine, conversation)
    # Stamp the most recent small-args arc (inside the protection window).
    response = engine.handle_tool_call("ctx_reduce", {"tags": "2"}, messages=conversation)
    payload = json.loads(response)
    assert payload["held"] == [2]
    engine.update_from_response({"prompt_tokens": 8_100})
    held_pass = run_pass(engine, conversation)
    # Still visible (held): prefix present, placeholder NOT applied.
    tool_texts = [m["content"] for m in held_pass if m.get("role") == "tool"]
    assert tool_texts[1].startswith("§2§ ")
    assert PLACEHOLDER.format(n=2) not in tool_texts


def test_mapping_mismatch_fails_open(make_engine):
    engine = make_engine()
    counter = Counter()
    conversation: list[dict] = []
    for _ in range(2):
        conversation.extend(counter.turn())
    request = build_request(conversation)
    broken = [request[0]] + request[2:]  # a history row vanished from the request
    assert engine.select_context(
        broken, conversation_messages=conversation, budget_tokens=0
    ) is None


def test_disabled_engine_passthrough(make_engine):
    engine = make_engine(enabled=False)
    counter = Counter()
    conversation: list[dict] = []
    for _ in range(2):
        conversation.extend(counter.turn())
    request = build_request(conversation)
    assert engine.select_context(
        request, conversation_messages=conversation, budget_tokens=0
    ) is None
    assert engine.get_tool_schemas() == []


def test_pass_failure_falls_back_to_lkg(make_engine, monkeypatch):
    engine = make_engine()
    counter = Counter()
    conversation: list[dict] = []
    for _ in range(3):
        conversation.extend(counter.turn())
    engine.update_from_response({"prompt_tokens": 12_000})
    good = run_pass(engine, conversation)

    import hce_plugin.engine as engine_module

    def boom(*_a, **_k):
        raise RuntimeError("simulated internal failure")

    monkeypatch.setattr(engine_module, "scan_conversation", boom)
    request = build_request(conversation)
    replayed = engine.select_context(
        request, conversation_messages=conversation, budget_tokens=0
    )
    assert replayed is not None
    assert fingerprint_list(replayed) == fingerprint_list(good)


def test_third_held_pass_spends_the_bust(make_engine):
    """连续第 3 趟挂住就放行：小会话永远到不了 2% 门槛，否则盖章永不落地。"""
    engine = make_engine()                      # 200k window => 2% gate = 4k tokens
    counter = Counter()
    conversation: list[dict] = []
    for _ in range(3):
        conversation.extend(counter.turn())
    conversation.extend(counter.big())
    conversation.extend(counter.turn())
    engine.update_from_response({"prompt_tokens": 20_000})
    out = run_pass(engine, conversation)
    small_tag = find_tag_for(out, "data")       # a tiny result, far below the gate
    payload = json.loads(engine.handle_tool_call(
        "ctx_reduce", {"tags": str(small_tag)}, messages=conversation
    ))
    assert payload["stamped"] == [small_tag]

    landed_on = None
    for pass_no in range(2, 6):
        engine.update_from_response({"prompt_tokens": 21_000})
        current = run_pass(engine, conversation)
        if PLACEHOLDER.format(n=small_tag) in [m.get("content") for m in current]:
            landed_on = pass_no
            break
    assert landed_on is not None, "held drops must eventually land"
    assert landed_on <= 4, landed_on            # 3 consecutive held passes, then a bust


def test_newest_tool_arcs_always_visible(make_engine):
    """结构性最小量：最新 3 个工具弧不被自动回收（窗口再大也一样）。"""
    engine = make_engine(context_length=1_000_000)
    counter = Counter()
    conversation: list[dict] = []
    for _ in range(4):
        conversation.extend(counter.turn())
    engine.update_from_response({"prompt_tokens": 1_000})
    out = run_pass(engine, conversation)
    newest_three = [find_tag_for(out, "data")][:0]
    tags = [int(m["content"].split(" ", 1)[0][1:-1]) for m in out if m.get("role") == "tool"]
    for tag in tags[-3:]:
        payload = json.loads(engine.handle_tool_call(
            "ctx_reduce", {"tags": str(tag)}, messages=conversation
        ))
        assert payload["held"] == [tag], (tag, payload)


def test_remove_downgrades_when_owner_would_empty(make_engine):
    """大参数弧本该整体移除；但若 owner 没了调用就只剩空壳，降级为骨架。"""
    engine = make_engine(context_length=200_000)
    counter = Counter()
    conversation: list[dict] = []
    for _ in range(2):
        conversation.extend(counter.turn())
    big_args = json.dumps({"path": "/data/", "blob": "a" * 2000})
    conversation.extend(counter.turn(assistant_text="", tool_args=big_args))  # empty shell
    for _ in range(3):
        conversation.extend(counter.turn())
    conversation.extend(counter.big())
    engine.update_from_response({"prompt_tokens": 140_000})
    out = run_pass(engine, conversation)
    target_call_id = next(
        call["id"] for m in conversation if m.get("role") == "assistant"
        for call in m.get("tool_calls", [])
        if "blob" in call.get("function", {}).get("arguments", "")
    )
    target = next(m for m in out if m.get("role") == "tool"
                  and m.get("tool_call_id") == target_call_id)
    tag = int(target["content"].split(" ", 1)[0][1:-1])

    engine.handle_tool_call("ctx_reduce", {"tags": str(tag)}, messages=conversation)
    engine.update_from_response({"prompt_tokens": 141_000})
    landed = run_pass(engine, conversation)
    assert PLACEHOLDER.format(n=tag) in [m.get("content") for m in landed]
    owner_index = next(
        i for i, m in enumerate(conversation) if m.get("role") == "assistant"
        and any(c.get("id") == target_call_id for c in m.get("tool_calls", []))
    )
    head_len = len(landed) - len(conversation)
    assert landed[owner_index + head_len].get("tool_calls"), "call must survive as a skeleton"


def test_lkg_dropped_when_model_changed(make_engine, monkeypatch):
    """快照跨模型不可重放：模型变了就丢掉快照（不是硬着头皮回放旧字节）。"""
    engine = make_engine()
    counter = Counter()
    conversation: list[dict] = []
    for _ in range(3):
        conversation.extend(counter.turn())
    engine.update_from_response({"prompt_tokens": 12_000})
    good = run_pass(engine, conversation)
    assert engine._store.load_lkg(engine._session_id) is not None

    import hce_plugin.engine as engine_module

    def boom(*_a, **_k):
        raise RuntimeError("simulated internal failure")

    engine.update_model("other-model", engine.context_length)
    monkeypatch.setattr(engine_module, "scan_conversation", boom)
    assert engine.select_context(
        build_request(conversation), conversation_messages=conversation, budget_tokens=0
    ) is None, "stale-model snapshot must not be replayed"
    assert engine._store.load_lkg(engine._session_id) is None
