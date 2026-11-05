"""compress() lanes: force-landing, oldest-first skeletonization, partial-head
safety, idempotence and the placeholder re-binding after a commit."""

from __future__ import annotations

import json

from host_sim import build_request, fingerprint_list, make_turn


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


def seed(make_engine, conversation, context_length=30_000):
    engine = make_engine(context_length=context_length)
    engine.update_from_response({"prompt_tokens": 10_000})
    output = engine.select_context(
        build_request(conversation), conversation_messages=conversation,
        budget_tokens=context_length,
    )
    assert output is not None
    return engine, output


def test_compress_force_lands_held_and_skeletonizes_oldest(make_engine):
    counter = Counter()
    conversation: list[dict] = []
    for _ in range(3):
        conversation.extend(counter.turn())
    conversation.extend(counter.big())       # oldest big -> skeletonized
    for _ in range(2):
        conversation.extend(counter.turn())
    conversation.extend(counter.big())       # newest big -> kept (target reached first)
    conversation.extend(counter.turn())
    engine, _out = seed(make_engine, conversation)

    # Queue the oldest arcs explicitly; they may be held by the window — compress
    # must land them regardless.
    engine.handle_tool_call("ctx_reduce", {"tags": "1,2,3,4"}, messages=conversation)
    compressed = engine.compress(conversation)
    assert compressed != conversation
    texts = [m.get("content") for m in compressed]
    assert "[dropped §4§]" in texts          # oldest big result replaced
    big_kept = [t for t in texts if isinstance(t, str) and "xxxxx" in t]
    assert big_kept, "newest big result must survive"
    assert engine.compression_count == 1

    # Second compress on the committed output: idempotent, no further growth.
    again = engine.compress(compressed)
    assert fingerprint_list(again) == fingerprint_list(compressed)
    assert engine.compression_count == 1

    # Owner tool_calls stay intact (skeleton keeps real arguments — real-or-absent).
    owners = [m for m in compressed if m.get("role") == "assistant" and m.get("tool_calls")]
    assert owners, "skeleton keeps the call"


def test_compress_noop_short_session(make_engine):
    counter = Counter()
    conversation: list[dict] = []
    for _ in range(2):
        conversation.extend(counter.turn())
    engine, _out = seed(make_engine, conversation, context_length=200_000)
    before_count = engine.compression_count
    compressed = engine.compress(conversation)
    assert fingerprint_list(compressed) == fingerprint_list(conversation)
    assert engine.compression_count == before_count


def test_compress_partial_head_never_compacts_unseen_tags(make_engine):
    counter = Counter()
    conversation: list[dict] = []
    for _ in range(2):
        conversation.extend(counter.turn())
    conversation.extend(counter.big())
    for _ in range(2):
        conversation.extend(counter.turn())
    engine, out = seed(make_engine, conversation, context_length=200_000)
    tail_tag_text = next(
        m["content"] for m in out if m.get("role") == "tool" and "xxxxx" in str(m.get("content"))
    )
    tail_tag = int(tail_tag_text.split(" ", 1)[0][1:-1])

    # Gateway manual /compress with --partial: engine.compress sees only the head.
    head = conversation[:4]
    compressed_head = engine.compress(head)
    assert isinstance(compressed_head, list)

    # The next full pass must still see the tail tag active with its prefix.
    engine.update_from_response({"prompt_tokens": 11_000})
    after = engine.select_context(
        build_request(conversation), conversation_messages=conversation,
        budget_tokens=engine.context_length,
    )
    assert after is not None
    still_tagged = any(
        m.get("role") == "tool" and isinstance(m.get("content"), str)
        and m["content"].startswith(f"§{tail_tag}§ ")
        for m in after
    )
    assert still_tagged


def test_compress_commit_then_pass_rebinds_placeholder(make_engine):
    counter = Counter()
    conversation: list[dict] = []
    for _ in range(3):
        conversation.extend(counter.turn())
    conversation.extend(counter.big())
    for _ in range(2):
        conversation.extend(counter.turn())
    conversation.extend(counter.big())
    for _ in range(1):
        conversation.extend(counter.turn())
    engine, out = seed(make_engine, conversation, context_length=30_000)
    target = next(
        (int(m["content"].split(" ", 1)[0][1:-1]) for m in out
         if m.get("role") == "tool" and str(m.get("content", "")).count("x") > 1000
         and m["content"].startswith("§4§")),
        None,
    )
    assert target is not None
    engine.handle_tool_call("ctx_reduce", {"tags": str(target)}, messages=conversation)
    compressed = engine.compress(conversation)
    placeholder = f"[dropped §{target}§]"
    assert placeholder in [m.get("content") for m in compressed]

    # Host commits the output as the new transcript; the next pass must recognize
    # the placeholder carrier (no fresh tag, no double prefix).
    max_before = engine._store.max_tag_number(engine._session_id)
    engine.update_from_response({"prompt_tokens": 12_000})
    after = engine.select_context(
        build_request(compressed), conversation_messages=compressed,
        budget_tokens=engine.context_length,
    )
    assert after is not None
    assert engine._store.max_tag_number(engine._session_id) == max_before
    assert not any(
        isinstance(m.get("content"), str) and m["content"].startswith("§") and placeholder in m["content"]
        for m in after
    )


def test_compress_never_raises_on_garbage(make_engine):
    engine = make_engine()
    garbage = [{"role": "tool"}, "not-a-dict", {"role": "user", "content": []}]
    out = engine.compress(garbage)
    assert isinstance(out, list)
