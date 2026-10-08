"""Tests for tailguard.openai_client with a fake client (no network)."""
import json

import httpx
import openai
import pytest

from tailguard import openai_client as oc


# --- fakes ---

class _Fn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _TC:
    def __init__(self, i, name, args):
        self.id = i
        self.type = "function"
        self.function = _Fn(name, args)


class _Msg:
    def __init__(self, content, tool_calls):
        self.content = content
        self.tool_calls = tool_calls


class _TLP:
    def __init__(self, token, logprob):
        self.token = token
        self.logprob = logprob


class _Content:
    def __init__(self, tlps):
        self.top_logprobs = tlps


class _LP:
    def __init__(self, tlps):
        self.content = [_Content(tlps)] if tlps is not None else None


class _Choice:
    def __init__(self, message, logprobs):
        self.message = message
        self.logprobs = logprobs


class _Det:
    def __init__(self, reasoning_tokens):
        self.reasoning_tokens = reasoning_tokens


class _Usage:
    def __init__(self, p=10, c=5, t=15, r=0):
        self.prompt_tokens = p
        self.completion_tokens = c
        self.total_tokens = t
        self.completion_tokens_details = _Det(r)


class FakeResp:
    def __init__(self, content="ok", tool_calls=None, tlps=None, usage=None):
        self._content = content
        self._tcs = tool_calls or []
        self._tlps = tlps
        self.usage = usage or _Usage()

    @property
    def choices(self):
        msg = _Msg(self._content,
                   [_TC(tc["id"], tc["name"], tc["arguments"]) for tc in self._tcs])
        tlps = None if self._tlps is None else [_TLP(t["token"], t["logprob"])
                                               for t in self._tlps]
        return [_Choice(msg, _LP(tlps))]

    def model_dump(self):
        return {"choices": [{"message":
                {"content": self._content,
                 "tool_calls": [{"id": tc["id"], "type": "function",
                                 "function": {"name": tc["name"],
                                              "arguments": tc["arguments"]}}
                                for tc in self._tcs]},
                "logprobs": ({"content": [{"top_logprobs": self._tlps}]}
                             if self._tlps is not None else None)}],
                "usage": {"prompt_tokens": self.usage.prompt_tokens,
                          "completion_tokens": self.usage.completion_tokens,
                          "total_tokens": self.usage.total_tokens,
                          "completion_tokens_details":
                          {"reasoning_tokens":
                           self.usage.completion_tokens_details.reasoning_tokens}}}


class FakeCompletions:
    def __init__(self, script):
        self.script = list(script)  # FakeResp or Exception instances
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, script):
        self.chat = type("C", (), {"completions": FakeCompletions(script)})()


def _err(cls, status):
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return cls("boom", response=httpx.Response(status, request=req), body=None)


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(oc, "CACHE_DIR", cache)
    monkeypatch.setattr(oc, "LEDGER_PATH", ledger)
    monkeypatch.setattr(oc.time, "sleep", lambda s: None)
    oc.set_client(None)
    yield cache, ledger
    oc.set_client(None)


def _caps(tmp_path, entry):
    p = tmp_path / "caps.json"
    p.write_text(json.dumps({"m": entry}))
    return p


# --- tests ---

def test_cache_hit_records_zero_tokens(dirs):
    cache, ledger = dirs
    oc.set_client(FakeClient([FakeResp(), FakeResp()]))
    msgs = [{"role": "user", "content": "hi"}]
    r1 = oc.raw_create("gpt-5.4-mini", msgs, None, {"max_completion_tokens": 5},
                       "t", 5)
    assert not isinstance(r1, dict)  # live object
    assert ledger.exists()
    n1 = len(ledger.read_text().splitlines())
    r2 = oc.raw_create("gpt-5.4-mini", msgs, None, {"max_completion_tokens": 5},
                       "t", 5)
    assert isinstance(r2, dict)  # cache hit
    assert len(ledger.read_text().splitlines()) == n1  # zero new tokens


def test_ledger_sums_by_pool_and_date(dirs):
    _, ledger = dirs
    oc.set_client(FakeClient([FakeResp(usage=_Usage(10, 5, 15)),
                              FakeResp(usage=_Usage(100, 50, 150))]))
    oc.raw_create("gpt-5.4-mini", [{"role": "user", "content": "a"}], None,
                  {"max_completion_tokens": 5}, "t", 5)
    oc.raw_create("gpt-5.4", [{"role": "user", "content": "b"}], None,
                  {"max_completion_tokens": 5}, "t", 5)
    assert oc.used_today("small") == 15
    assert oc.used_today("large") == 150
    assert oc.used_today("small", utc_date="2000-01-01") == 0


def test_budget_exceeded(dirs, monkeypatch):
    cache, ledger = dirs
    monkeypatch.setattr(oc, "DAILY_CAP", {"small": 1, "large": 1})
    oc.set_client(FakeClient([FakeResp()]))
    with pytest.raises(oc.BudgetExceeded):
        oc.raw_create("gpt-5.4-mini", [{"role": "user", "content": "a" * 500}],
                      None, {"max_completion_tokens": 5}, "t", 5)
    assert not ledger.exists()
    assert list(cache.glob("*.json")) == []


def test_pool_of_unknown_raises():
    with pytest.raises(ValueError):
        oc.pool_of("not-a-model")


def test_build_params_drops_unaccepted_keys(tmp_path):
    entry = {"max_completion_tokens": True, "max_tokens": False,
             "temperature": False, "logprobs": False, "tools": False,
             "reasoning_effort": []}
    p = _caps(tmp_path, entry)
    params = oc.build_params("m", "judge", 7, want_logprobs=True, caps_path=p)
    assert params == {"max_completion_tokens": 7}
    entry2 = dict(entry, temperature=True, logprobs=True,
                  reasoning_effort=["low", "minimal"])
    p2 = _caps(tmp_path, entry2)
    params2 = oc.build_params("m", "judge", 7, want_logprobs=True, caps_path=p2)
    assert params2 == {"max_completion_tokens": 7, "temperature": 0,
                       "logprobs": True, "top_logprobs": 5,
                       "reasoning_effort": "minimal"}  # lowest accepted level


def test_retry_then_success(dirs):
    _, ledger = dirs
    oc.set_client(FakeClient([_err(openai.RateLimitError, 429), FakeResp()]))
    resp = oc.raw_create("gpt-5.4-mini", [{"role": "user", "content": "a"}],
                         None, {"max_completion_tokens": 5}, "t", 5)
    assert oc.get_message(resp)["content"] == "ok"
    assert len(ledger.read_text().splitlines()) == 1


def test_bad_request_no_retry(dirs):
    _, ledger = dirs
    fake = FakeClient([_err(openai.BadRequestError, 400)])
    oc.set_client(fake)
    with pytest.raises(openai.BadRequestError):
        oc.raw_create("gpt-5.4-mini", [{"role": "user", "content": "a"}],
                      None, {"max_completion_tokens": 5}, "t", 5)
    assert len(fake.chat.completions.calls) == 1
    assert not ledger.exists()


def test_adapter_live_and_cached_agree():
    live = FakeResp(content="yes", tlps=[{"token": "yes", "logprob": -0.1}])
    cached = live.model_dump()
    assert oc.get_usage(live) == oc.get_usage(cached)
    assert oc.get_message(live) == oc.get_message(cached)
    assert oc.get_top_logprobs(live) == oc.get_top_logprobs(cached)
    assert oc.get_usage(live)["reasoning_tokens"] == 0


def test_call_passes_tool_choice_override(dirs, tmp_path, monkeypatch):
    p = tmp_path / "caps.json"
    p.write_text(json.dumps({"gpt-5.4-mini":
                             {"max_completion_tokens": True, "temperature": True,
                              "logprobs": False, "reasoning_effort": []}}))
    monkeypatch.setattr(oc, "CAPS_PATH", p)
    fake = FakeClient([FakeResp()])
    oc.set_client(fake)
    oc.call("gpt-5.4-mini", [{"role": "user", "content": "a"}],
            tools=[{"type": "function"}], purpose="frontier",
            max_out=9, tool_choice="auto")
    kw = fake.chat.completions.calls[0]
    assert kw["tool_choice"] == "auto"
    assert kw["max_completion_tokens"] == 9
