"""Reduction nudge — a tail-only reminder that tells the model what is reclaimable.

Why this exists (measured, not assumed): with tags live but no reminder, the model
never volunteers ``ctx_reduce``. Seven days of production data on a 1M-window model
produced 281 tags and zero stamps — the M0 thesis "the model will stamp what it no
longer needs" is false without a prompt to do it.

Design constraints (measured, not assumed):

- **Advisory, not directive** — a reminder that shouts gets ignored or over-followed;
  the copy escalates only with measured severity.
- **Scale against the tail, not the window** — severity is ``U/T``: reclaimable mass
  over the tail's own content mass. The window never enters, because a modest pile on
  a huge window is still a dirty tail; anchoring to the window instead makes the
  reminder unreachable far below the provider limit.
- **Ignore what cannot be dropped yet** — protected-window content is excluded from
  both the count and the token figure, so the reminder never asks for the impossible.
- **Never nag after the agent complied** — no reminder on the pass that just landed
  drops, and re-fires need real regrowth (25k or 8% of the tail, whichever is larger).
- **Quiet floors** — under 60k of tail or 25k of reclaimable mass the reminder is
  noise at any ratio.
- **Tail-only bytes** — the reminder is appended to the last message and stripped
  again only on passes that may rewrite history, so it can never mutate a cached
  prefix on a derived pass.
"""

from __future__ import annotations

import re
from typing import Optional

NUDGE_OPEN = "<system-reminder>"
NUDGE_CLOSE = "</system-reminder>"

# Severity bands: reclaimable tokens / tail content mass (U/T).
S_GENTLE = 0.20
S_FIRM = 0.40
S_URGENT = 0.60
LEVEL_RANK = {"gentle": 1, "firm": 2, "urgent": 3}

# Quiet floors: below either, no ratio makes a reminder worth its tail bytes.
QUIET_TAIL_TOKENS = 60_000
QUIET_RECLAIMABLE_TOKENS = 25_000
# Re-fire inside a band needs this much new reclaimable mass.
REFIRE_TAIL_FRACTION = 0.08
REFIRE_FLOOR_TOKENS = 25_000
HINT_LIMIT = 4

_REMINDER_RE = re.compile(
    r"\n*" + re.escape(NUDGE_OPEN) + r".*?" + re.escape(NUDGE_CLOSE) + r"\n*",
    re.DOTALL,
)


def strip_reminders(text: str) -> str:
    """Remove every reminder span we previously injected (idempotent)."""
    if not text or NUDGE_OPEN not in text:
        return text
    cleaned = _REMINDER_RE.sub("\n", text)
    return cleaned.strip("\n") if cleaned != text else cleaned


def band_for(severity: float) -> Optional[str]:
    if severity >= S_URGENT:
        return "urgent"
    if severity >= S_FIRM:
        return "firm"
    if severity >= S_GENTLE:
        return "gentle"
    return None


def _human_tokens(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{int(round(value / 1000))}k"
    return str(int(value))


_BODIES = {
    "gentle": (
        "Housekeeping: {n} spent tool outputs (~{tokens} tokens) are reclaimable. "
        "Stamp one with ctx_reduce as soon as you have taken what you need from it — "
        "no need to wait for the task to finish. This queues the drop; the original "
        "stays retrievable with ctx_expand."
    ),
    "firm": (
        "{n} spent tool outputs (~{tokens} tokens) are reclaimable. Make a ctx_reduce "
        "pass over the ones you have already used, then continue."
    ),
    "urgent": (
        "{n} spent tool outputs (~{tokens} tokens) are still unstamped. Call "
        "ctx_reduce now, before your next tool call, on every output you have "
        "already used."
    ),
}


def render_nudge(
    level: str,
    count: int,
    tokens: int,
    oldest: list[tuple[int, Optional[str]]],
) -> str:
    """The reminder text (without surrounding blank lines)."""
    body = _BODIES[level].format(n=count, tokens=_human_tokens(tokens))
    if oldest:
        listed = " · ".join(
            f"§{number}§{(' ' + name) if name else ''}" for number, name in oldest[:HINT_LIMIT]
        )
        body += f" Oldest reclaimable: {listed}."
    return f"{NUDGE_OPEN}\n{body}\n{NUDGE_CLOSE}"


def decide(
    *,
    reclaimable_tokens: int,
    reclaimable_count: int,
    tail_tokens: int,
    oldest: list[tuple[int, Optional[str]]],
    last_fired_tokens: Optional[int],
    last_level: Optional[str],
    just_landed: bool,
) -> Optional[tuple[str, str]]:
    """Return ``(level, text)`` when a reminder should ride this pass, else ``None``.

    ``tail_tokens`` is ``T``: the content mass of the tail the provider sees, with
    reasoning and already-dropped payloads excluded. ``last_fired_tokens``/
    ``last_level`` are the persisted hysteresis state: a band escalation always fires;
    staying in a band needs real regrowth.
    """
    if just_landed or reclaimable_count <= 0 or tail_tokens <= 0:
        return None
    if tail_tokens < QUIET_TAIL_TOKENS or reclaimable_tokens < QUIET_RECLAIMABLE_TOKENS:
        return None
    level = band_for(reclaimable_tokens / tail_tokens)
    if level is None:
        return None

    if last_fired_tokens is None:
        return level, render_nudge(level, reclaimable_count, reclaimable_tokens, oldest)

    previous_rank = LEVEL_RANK.get(last_level or "", 0)
    if LEVEL_RANK[level] > previous_rank:
        return level, render_nudge(level, reclaimable_count, reclaimable_tokens, oldest)

    growth = reclaimable_tokens - int(last_fired_tokens)
    threshold = max(
        REFIRE_FLOOR_TOKENS, int(round(REFIRE_TAIL_FRACTION * tail_tokens))
    )
    if growth >= threshold:
        return level, render_nudge(level, reclaimable_count, reclaimable_tokens, oldest)
    return None
