"""Accuracy metrics: character / word error rate."""

from __future__ import annotations


def _lev(a: list | str, b: list | str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _norm(s: str) -> str:
    return " ".join(s.split())


def cer(truth: str, hyp: str) -> float:
    """Character error rate (whitespace-normalised). 0.001 == 0.1 %."""
    t, h = _norm(truth), _norm(hyp)
    return _lev(t, h) / max(1, len(t))


def wer(truth: str, hyp: str) -> float:
    t, h = truth.split(), hyp.split()
    return _lev(t, h) / max(1, len(t))
