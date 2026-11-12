"""Reduction nudge — a tail-only reminder that tells the model what is reclaimable.

Why this exists (measured, not assumed): with tags live but no reminder, the model
never volunteers ``ctx_reduce``. Seven days of production data on a 1M-window model
produced 281 tags and zero stamps — the M0 thesis "the model will stamp what it no
longer needs" is false without a prompt to do it.

Design constraints taken from the reference implementation's shipped lessons:

- **Advisory, not directive** — a reminder that shouts gets ignored or over-followed;
  the copy escalates only with measured severity.
- **Scale against the working window, not the raw pile** — a modest pile on a huge
  window is not urgent. Severity = reclaimable/usable, with bands at 20/40/60%.
- **Ignore what cannot be dropped yet** — protected-window content is excluded from
  both the count and the token figure, so the reminder never asks for the impossible.
- **Never nag after the agent complied** — no reminder on the pass that just landed
  drops, and re-fires need real regrowth (8% of the last figure, with a floor).
- **Tail-only bytes** — the reminder is appended to the last message and stripped
  again only on passes that may rewrite history, so it can never mutate a cached
  prefix on a derived pass.
"""

from __future__ import annotations

import re
from typing import Optional

NUDGE_OPEN = "<system-reminder>"
NUDGE_CLOSE = "</system-reminder>"

# Severity bands (reclaimable tokens / usable window).
S_GENTLE = 0.20
S_FIRM = 0.40
S_URGENT = 0.60
LEVEL_RANK = {"gentle": 1, "firm": 2, "urgent": 3}

# Absolute floor: below this the reminder is noise on any window size.
REFIRE_FRACTION = 0.08
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


def floor_tokens(usable: float) -> int:
    """Smallest reclaimable mass worth mentioning, scaled down for small windows."""
    return int(min(25_000, max(2_000, round(0.02 * max(usable, 0)))))


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
    usable: float,
    oldest: list[tuple[int, Optional[str]]],
    last_fired_tokens: Optional[int],
    last_level: Optional[str],
    just_landed: bool,
) -> Optional[tuple[str, str]]:
    """Return ``(level, text)`` when a reminder should ride this pass, else ``None``.

    ``last_fired_tokens``/``last_level`` are the persisted hysteresis state: a band
    escalation always fires; staying in a band needs real regrowth.
    """
    if just_landed or reclaimable_count <= 0 or usable <= 0:
        return None
    severity = reclaimable_tokens / usable
    level = band_for(severity)
    if level is None:
        return None
    floor = floor_tokens(usable)
    if reclaimable_tokens < floor:
        return None

    if last_fired_tokens is None:
        return level, render_nudge(level, reclaimable_count, reclaimable_tokens, oldest)

    previous_rank = LEVEL_RANK.get(last_level or "", 0)
    if LEVEL_RANK[level] > previous_rank:
        return level, render_nudge(level, reclaimable_count, reclaimable_tokens, oldest)

    growth = reclaimable_tokens - int(last_fired_tokens)
    threshold = max(floor, int(round(REFIRE_FRACTION * max(int(last_fired_tokens), 0))))
    if growth >= threshold:
        return level, render_nudge(level, reclaimable_count, reclaimable_tokens, oldest)
    return None
