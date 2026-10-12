"""
Page ordering.

Filenames like "1.jpg", "2.jpg", ..., "10.jpg" must NOT be sorted as plain
strings — a lexical sort puts "10" before "2". This does a natural sort
(split into text/number runs, compare number runs numerically) instead.

This is only used to propose an initial order on upload. It's a heuristic,
not ground truth — the UI always lets the user confirm/reorder before
translation runs, since filenames aren't guaranteed reliable.
"""
from __future__ import annotations

import re

_NUM_RE = re.compile(r"(\d+)")


def natural_sort_key(filename: str) -> tuple:
    parts = _NUM_RE.split(filename)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def natural_sort(filenames: list[str]) -> list[str]:
    return sorted(filenames, key=natural_sort_key)
