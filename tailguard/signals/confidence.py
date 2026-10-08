"""Confidence helpers (spec P3.3 value-token span; P5.1 features come later).

The span function is pure (takes a decode callable) so it is unit-testable
without a GPU; run_specialist.py passes the real tokenizer's decode.
"""
from __future__ import annotations


def find_value_token_idx(text: str, gen_ids: list[int], decode, value: str
                         ) -> tuple[list[int], bool]:
    """Token indices whose char span overlaps the value string's span.

    Char end of generated token i: len(decode(gen_ids[:i+1])); token i covers
    [ends[i-1], ends[i]) with ends[-1] = 0. Returns (indices, found).
    """
    if not value:
        return [], False
    start = text.find(value)
    if start == -1:
        return [], False
    end = start + len(value)
    ends = [len(decode(gen_ids[: i + 1])) for i in range(len(gen_ids))]
    idx = []
    for i in range(len(gen_ids)):
        lo = ends[i - 1] if i > 0 else 0
        hi = ends[i]
        if lo < end and hi > start:
            idx.append(i)
    return idx, True
