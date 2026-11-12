"""M1 reduction nudge: fires on measured reclaimable mass, is tail-only, and never
nags after the agent complies."""

from __future__ import annotations

import json

from hce_plugin import nudge as nudge_mod
from hce_plugin.nudge import NUDGE_OPEN, decide, render_nudge, strip_reminders
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
                         tool_args=json.dumps({"path": f"/data/f{self.n}.bin"}),
                         **kwargs)


def run_pass(engine, conversation):
    output = engine.select_context(
        build_request(conversation), conversation_messages=conversation,
        budget_tokens=engine.context_length,
    )
    assert output is not None
    return output


def user_texts(output):
    return [m.get("content") for m in output if m.get("role") == "user"]


# ── 纯函数契约 ───────────────────────────────────────────────────────────

def test_strip_reminders_is_idempotent_and_targeted():
    text = "hello\n\n<system-reminder>\nstamp these\n</system-reminder>"
    assert strip_reminders(text) == "hello"
    assert strip_reminders(strip_reminders(text)) == "hello"
    assert strip_reminders("plain text") == "plain text"
    # foreign markup is untouched
    other = "keep <system>this</system>"
    assert strip_reminders(other) == other


def test_levels_escalate_with_severity():
    assert nudge_mod.band_for(0.19) is None
    assert nudge_mod.band_for(0.20) == "gentle"
    assert nudge_mod.band_for(0.45) == "firm"
    assert nudge_mod.band_for(0.75) == "urgent"
    text = render_nudge("gentle", 3, 12_000, [(7, "bash")])
    assert text.startswith(NUDGE_OPEN) and text.endswith("</system-reminder>")
    assert "§7§ bash" in text and "12k" in text


def test_decide_needs_floor_and_regrowth():
    kwargs = dict(reclaimable_count=5, usable=200_000, oldest=[(1, "bash")],
                  just_landed=False)
    # below the 2% floor (4k) -> silent
    assert decide(reclaimable_tokens=3_000, last_fired_tokens=None,
                  last_level=None, **kwargs) is None
    # above floor, first fire
    first = decide(reclaimable_tokens=50_000, last_fired_tokens=None,
                   last_level=None, **kwargs)
    assert first is not None and first[0] == "gentle"
    # same band, tiny regrowth -> silent
    assert decide(reclaimable_tokens=52_000, last_fired_tokens=50_000,
                  last_level="gentle", **kwargs) is None
    # same band, real regrowth (>= max(floor, 8%)) -> fires
    assert decide(reclaimable_tokens=60_000, last_fired_tokens=50_000,
                  last_level="gentle", **kwargs) is not None
    # band escalation always fires
    escalated = decide(reclaimable_tokens=100_000, last_fired_tokens=98_000,
                       last_level="gentle", **kwargs)
    assert escalated is not None and escalated[0] == "firm"
    # the pass that lands drops never reminds
    assert decide(reclaimable_tokens=200_000, last_fired_tokens=None,
                  last_level=None, just_landed=True,
                  reclaimable_count=5, usable=200_000, oldest=[(1, "bash")]) is None


# ── 端到端 ───────────────────────────────────────────────────────────────

def pile(counter: Counter, blocks: int = 6) -> list[dict]:
    """Enough big tool results that most of them sit outside the protection window."""
    rows: list[dict] = []
    for _ in range(blocks):
        rows.extend(counter.big())
        rows.extend(counter.turn())
    return rows


def test_nudge_rides_only_the_last_user_message(make_engine):
    engine = make_engine(context_length=200_000)
    counter = Counter()
    conversation: list[dict] = []
    conversation.extend(counter.turn())
    conversation.extend(pile(counter))
    engine.update_from_response({"prompt_tokens": 40_000})
    output = run_pass(engine, conversation)
    texts = user_texts(output)
    assert any(NUDGE_OPEN in str(t) for t in texts), "a pile this big must be mentioned"
    assert NUDGE_OPEN in str(texts[-1]), "only the tail carries it"
    assert all(NUDGE_OPEN not in str(t) for t in texts[:-1])

    # Frozen history keeps the reminder byte-identical on a derived pass.
    conversation.extend(counter.turn())
    engine.update_from_response({"prompt_tokens": 40_500})
    second = run_pass(engine, conversation)
    assert user_texts(second)[:-1] == texts


def test_reminder_stripped_on_rebuild_pass(make_engine):
    engine = make_engine(context_length=200_000)
    counter = Counter()
    conversation: list[dict] = []
    conversation.extend(counter.turn())
    conversation.extend(pile(counter))
    engine.update_from_response({"prompt_tokens": 40_000})
    first = run_pass(engine, conversation)
    assert any(NUDGE_OPEN in str(m.get("content")) for m in first)

    # Cross the execute threshold: history may be rewritten, stale reminders go away.
    engine.update_from_response({"prompt_tokens": 150_000})
    conversation.extend(counter.turn())
    rebuilt = run_pass(engine, conversation)
    middle = [m.get("content") for m in rebuilt if m.get("role") == "user"][:-1]
    assert all(NUDGE_OPEN not in str(text) for text in middle)


def test_no_nudge_below_floor(make_engine):
    engine = make_engine(context_length=1_000_000)   # 2% floor = 20k tokens
    counter = Counter()
    conversation: list[dict] = []
    for _ in range(2):
        conversation.extend(counter.turn())
    engine.update_from_response({"prompt_tokens": 5_000})
    output = run_pass(engine, conversation)
    assert all(NUDGE_OPEN not in str(text) for text in user_texts(output))


def test_nudge_counters_recorded(make_engine):
    engine = make_engine(context_length=200_000)
    counter = Counter()
    conversation: list[dict] = []
    conversation.extend(counter.turn())
    conversation.extend(pile(counter))
    engine.update_from_response({"prompt_tokens": 40_000})
    run_pass(engine, conversation)
    assert int(engine._store.get_meta(engine._session_id, "nudges", 0) or 0) >= 1
    assert engine._store.get_meta(engine._session_id, "nudge_level") in {"gentle", "firm", "urgent"}
