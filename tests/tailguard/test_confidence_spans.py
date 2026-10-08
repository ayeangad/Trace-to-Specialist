"""Unit test for the P3.3 value-token span (GPU-free, tokenizer-only).

Loads only the Qwen2.5-3B-Instruct tokenizer; skips if HF is unreachable.
"""
import pytest

from tailguard.config import BASE_MODEL
from tailguard.signals.confidence import find_value_token_idx


@pytest.fixture(scope="module")
def tok():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    except Exception as e:
        pytest.skip(f"no HF access for tokenizer test: {e}")


def test_span_tokens_contain_value(tok):
    v = "$12,345 million"
    text = (f"<tool_call><function=submit_answer><parameter=value>{v}</parameter>"
            f"<parameter=filing_id>ACME-10K-2024</parameter></function></tool_call>")
    ids = tok(text, add_special_tokens=False)["input_ids"]
    idx, found = find_value_token_idx(text, ids, tok.decode, v)
    assert found and idx
    assert v in tok.decode([ids[i] for i in idx])


def test_span_missing_value(tok):
    text = "<tool_call><function=search_filing><parameter=query>x</parameter></function></tool_call>"
    ids = tok(text, add_special_tokens=False)["input_ids"]
    idx, found = find_value_token_idx(text, ids, tok.decode, "$9,999 million")
    assert not found and idx == []


def test_span_empty_value(tok):
    assert find_value_token_idx("abc", [1, 2, 3], lambda x: "abc", "") == ([], False)
