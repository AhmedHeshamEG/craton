"""Checks that only make sense for many tokens together.

A single token can be perfect while a file of them is broken: out of time order, or holding the
same measurement twice. Token files are written in time order and deduplicated, so readers can
stream them and a repeated measurement can't count double in training or, worse, leak into a
test set under a second name.
"""

from __future__ import annotations

from collections.abc import Iterable

from craton.spec.token import Problem, Token


def sequence_problems(tokens: Iterable[Token]) -> list[Problem]:
    """Time order (non-decreasing) and duplicates, for tokens meant to share one file."""
    problems: list[Problem] = []
    seen: dict[str, int] = {}
    last: float | None = None
    for i, tok in enumerate(tokens):
        if last is not None and tok.time < last:
            problems.append(
                Problem(
                    code="time_order",
                    field=f"tokens[{i}].time",
                    message=f"{tok.time} comes after {last}; token files are in time order",
                )
            )
        last = tok.time if last is None else max(last, tok.time)
        h = tok.token_hash()
        if h in seen:
            problems.append(
                Problem(
                    code="duplicate",
                    field=f"tokens[{i}]",
                    message=f"same token as tokens[{seen[h]}]",
                )
            )
        else:
            seen[h] = i
    return problems
