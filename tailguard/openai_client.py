"""Caps-aware, budget-safe OpenAI Chat Completions client.

Rules (spec P0.5):
- Chat Completions only: client.chat.completions.create(**params).
- Disk cache keyed by sha256 of {model, messages, tools, params}; cache hits
  return the cached dict and record zero tokens.
- Ledger: one line per real (non-cached) call in openai_ledger.jsonl.
- Budget: pre-call estimate vs DAILY_CAP per pool; raise BudgetExceeded.
- Retries: RateLimitError / 5xx APIStatusError up to 5 retries with backoff;
  BadRequestError never retried.
- build_params returns only keys accepted per docs/tailguard/openai_caps.json.

Response adapter: get_usage / get_message / get_top_logprobs accept both live
SDK objects and cached dicts, so agent loops work on either.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import openai
from dotenv import load_dotenv

load_dotenv()

from .config import (DAILY_CAP, EXP, OPENAI_FRONTIER_MODEL, OPENAI_JUDGE_MODEL,
                     OPENAI_STRONG_MODEL, POOL_LARGE, POOL_SMALL, ROOT)
from .logging_utils import append_jsonl, read_jsonl

CACHE_DIR = EXP / "cache"
LEDGER_PATH = EXP / "openai_ledger.jsonl"
CAPS_PATH = ROOT / "docs" / "tailguard" / "openai_caps.json"

# Preference order for reasoning effort: lowest first.
REASONING_LEVELS = ["none", "minimal", "low"]


class BudgetExceeded(Exception):
    """Raised when a request would cross the daily token cap. Callers catch,
    log, and exit cleanly (runs are resumable)."""


def pool_of(model: str) -> str:
    """Return 'small'/'large' pool for a model, else raise."""
    if model in POOL_SMALL:
        return "small"
    if model in POOL_LARGE:
        return "large"
    raise ValueError(f"model {model!r} is in neither POOL_SMALL nor POOL_LARGE")


def read_caps(caps_path: str | Path | None = None) -> dict:
    p = Path(caps_path) if caps_path else CAPS_PATH
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def build_params(model: str, purpose: str, max_out: int,
                 want_logprobs: bool = False,
                 caps_path: str | Path | None = None) -> dict:
    """Return only API params the model accepts (per openai_caps.json).

    - max budget key: max_completion_tokens if accepted, else max_tokens.
    - temperature=0 only if accepted.
    - logprobs/top_logprobs only if wanted AND accepted.
    - reasoning_effort only if accepted (lowest accepted level).
    """
    _ = purpose  # reserved for per-purpose tuning; caps decide today.
    caps = read_caps(caps_path).get(model, {})
    params: dict = {}
    if caps.get("max_completion_tokens"):
        params["max_completion_tokens"] = max_out
    elif caps.get("max_tokens"):
        params["max_tokens"] = max_out
    else:
        # Pre-probe default (the probe bypasses build_params with explicit
        # keys, so this path only fires before caps exist).
        params["max_completion_tokens"] = max_out
    if caps.get("temperature"):
        params["temperature"] = 0
    if want_logprobs and caps.get("logprobs"):
        params["logprobs"] = True
        params["top_logprobs"] = 5
    accepted_levels = caps.get("reasoning_effort") or []
    for level in REASONING_LEVELS:
        if level in accepted_levels:
            params["reasoning_effort"] = level
            break
    return params


def _jsonable(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return str(obj)


def request_key(model: str, messages: list, tools, params: dict) -> str:
    blob = json.dumps({"model": model, "messages": _jsonable(messages),
                       "tools": _jsonable(tools), "params": _jsonable(params)},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def cache_path(key: str, cache_dir: str | Path | None = None) -> Path:
    return Path(cache_dir or CACHE_DIR) / f"{key}.json"


def get_usage(resp) -> dict:
    """Token counts from a live SDK response or a cached dict."""
    if isinstance(resp, dict):
        u = (resp.get("usage") or {})
        det = u.get("completion_tokens_details") or {}
        rt = det.get("reasoning_tokens") if isinstance(det, dict) else None
    else:
        u = resp.usage
        det = getattr(u, "completion_tokens_details", None)
        rt = getattr(det, "reasoning_tokens", None) if det is not None else None
        u = {"prompt_tokens": u.prompt_tokens,
             "completion_tokens": u.completion_tokens,
             "total_tokens": u.total_tokens}
    return {"prompt_tokens": u.get("prompt_tokens", 0),
            "completion_tokens": u.get("completion_tokens", 0),
            "reasoning_tokens": rt if rt is not None else 0,
            "total_tokens": u.get("total_tokens", 0)}


def get_message(resp) -> dict:
    """{'content': str|None, 'tool_calls': [{'id','name','arguments'}]}."""
    if isinstance(resp, dict):
        m = resp["choices"][0]["message"]
        tcs = m.get("tool_calls") or []
        calls = [{"id": tc.get("id"), "name": (tc.get("function") or {}).get("name"),
                  "arguments": (tc.get("function") or {}).get("arguments", "")}
                 for tc in tcs]
        return {"content": m.get("content"), "tool_calls": calls}
    m = resp.choices[0].message
    calls = [{"id": tc.id, "name": tc.function.name,
              "arguments": tc.function.arguments or ""}
             for tc in (m.tool_calls or [])]
    return {"content": m.content, "tool_calls": calls}


def get_top_logprobs(resp) -> list[dict] | None:
    """[{'token','logprob'}] for the first content position, else None."""
    try:
        if isinstance(resp, dict):
            tls = resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            return [{"token": t["token"], "logprob": t["logprob"]} for t in tls]
        tls = resp.choices[0].logprobs.content[0].top_logprobs
        return [{"token": t.token, "logprob": t.logprob} for t in tls]
    except (KeyError, IndexError, TypeError, AttributeError):
        return None


def utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def used_today(pool: str, ledger_path: str | Path | None = None,
               utc_date: str | None = None) -> int:
    """Sum of total_tokens in the ledger for this UTC date + pool."""
    day = utc_date or utc_today()
    total = 0
    for row in read_jsonl(ledger_path or LEDGER_PATH):
        if row.get("utc_date") == day and row.get("pool") == pool:
            total += int(row.get("total_tokens", 0))
    return total


def estimate_tokens(messages: list, tools, max_out: int) -> int:
    return len(json.dumps(_jsonable(messages)) + json.dumps(_jsonable(tools))) // 3 + max_out


_client = None


def get_client():
    global _client
    if _client is None:
        _client = openai.OpenAI()
    return _client


def set_client(c) -> None:
    """Inject a fake client (tests)."""
    global _client
    _client = c


def raw_create(model: str, messages: list, tools, params: dict, purpose: str,
               max_out: int, ledger_path: str | Path | None = None,
               cache_dir: str | Path | None = None) -> object | dict:
    """Cache + budget + retries around chat.completions.create.

    Returns the live SDK response on a real call, or the cached dict on hit.
    """
    cdir = Path(cache_dir or CACHE_DIR)
    cdir.mkdir(parents=True, exist_ok=True)
    key = request_key(model, messages, tools, params)
    cp = cache_path(key, cdir)
    if cp.exists():
        return json.loads(cp.read_text())

    pool = pool_of(model)  # raises for unknown models before any spend
    if used_today(pool, ledger_path) + estimate_tokens(messages, tools, max_out) > DAILY_CAP[pool]:
        raise BudgetExceeded(
            f"pool={pool} used={used_today(pool, ledger_path)} "
            f"estimate={estimate_tokens(messages, tools, max_out)} "
            f"cap={DAILY_CAP[pool]}")

    kwargs = {"model": model, "messages": messages}
    if tools is not None:
        kwargs["tools"] = tools
    kwargs.update(params)

    last_err = None
    for attempt in range(6):  # initial try + up to 5 retries
        try:
            resp = get_client().chat.completions.create(**kwargs)
        except openai.BadRequestError:
            raise
        except openai.RateLimitError as e:
            last_err = e
        except openai.APIStatusError as e:
            if getattr(e, "status_code", 0) is not None and e.status_code >= 500:
                last_err = e
            else:
                raise
        else:
            cp.write_text(json.dumps(resp.model_dump()))
            u = get_usage(resp)
            append_jsonl(ledger_path or LEDGER_PATH,
                         {"utc_date": utc_today(), "model": model, "pool": pool,
                          "prompt_tokens": u["prompt_tokens"],
                          "completion_tokens": u["completion_tokens"],
                          "reasoning_tokens": u["reasoning_tokens"],
                          "total_tokens": u["total_tokens"],
                          "purpose": purpose, "cache_key": key})
            return resp
        time.sleep(min(60, 2 ** attempt + random.random()))
    raise last_err


def call(model: str, messages: list, tools=None, purpose: str = "",
         max_out: int = 512, want_logprobs: bool = False, **overrides):
    """Caps-aware call: build_params + raw_create. Overrides win (e.g.
    tool_choice='auto')."""
    params = build_params(model, purpose, max_out, want_logprobs)
    params.update(overrides)
    return raw_create(model, messages, tools, params, purpose, max_out)


# --- P0.6 capability probe ---

PROBE_MODELS = [OPENAI_FRONTIER_MODEL, OPENAI_JUDGE_MODEL, OPENAI_STRONG_MODEL,
                "gpt-5.4-nano", "gpt-4.1-nano"]
PROBE_MSG = [{"role": "user", "content": "Reply with the single word: ok"}]


def _try(model: str, messages: list, tools, params: dict, purpose: str = "probe"):
    """Returns (resp_or_None, error_str_or_None). Never raises OpenAI errors."""
    try:
        return raw_create(model, messages, tools, params, purpose,
                          params.get("max_completion_tokens",
                                     params.get("max_tokens", 16))), None
    except openai.OpenAIError as e:
        return None, f"{type(e).__name__}: {str(e)[:300]}"


def run_probe(out_path: str | Path | None = None) -> dict:
    """Probe each model; write docs/tailguard/openai_caps.json. Cost <5k tokens."""
    import sys
    sys.path.insert(0, str(ROOT))
    from environment.tools import TOOL_SCHEMAS

    caps: dict = {}
    for model in PROBE_MODELS:
        entry = {"max_completion_tokens": False, "max_tokens": False,
                 "temperature": False, "logprobs": False, "tools": False,
                 "reasoning_effort": [], "errors": {}}
        try:
            pool_of(model)
        except ValueError as e:
            entry["errors"]["model"] = str(e)
            caps[model] = entry
            continue
        # 1-2: max-output keys (test 1 first; test 2 independently).
        for key in ("max_completion_tokens", "max_tokens"):
            resp, err = _try(model, PROBE_MSG, None, {key: 16})
            if resp is not None:
                entry[key] = True
            else:
                entry["errors"][key] = err
                if err and err.startswith("NotFoundError"):
                    break
        out_key = "max_completion_tokens" if entry["max_completion_tokens"] else "max_tokens"
        base = {out_key: 16}
        # 3: temperature.
        resp, err = _try(model, PROBE_MSG, None, {**base, "temperature": 0})
        if resp is not None:
            entry["temperature"] = True
        else:
            entry["errors"]["temperature"] = err
        # 4: logprobs.
        resp, err = _try(model, PROBE_MSG, None,
                         {**base, "logprobs": True, "top_logprobs": 5})
        if resp is not None:
            tls = get_top_logprobs(resp)
            entry["logprobs"] = bool(tls) and all(
                "token" in t and "logprob" in t for t in tls)
            if not entry["logprobs"]:
                entry["errors"]["logprobs"] = "accepted but top_logprobs unusable"
        else:
            entry["errors"]["logprobs"] = err
        # 5: tools.
        resp, err = _try(model,
                         [{"role": "user", "content": "Search for ACME 2024 using the tool."}],
                         [TOOL_SCHEMAS[0]], base)
        if resp is not None:
            entry["tools"] = bool(get_message(resp)["tool_calls"])
            if not entry["tools"]:
                entry["errors"]["tools"] = "accepted but no tool_calls returned"
        else:
            entry["errors"]["tools"] = err
        # 6: reasoning_effort levels.
        for level in REASONING_LEVELS:
            resp, err = _try(model, PROBE_MSG, None,
                             {**base, "reasoning_effort": level})
            if resp is not None:
                entry["reasoning_effort"].append(level)
            else:
                entry["errors"][f"reasoning_effort={level}"] = err
        caps[model] = entry
    out = Path(out_path) if out_path else CAPS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(caps, indent=1) + "\n")
    return caps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    if a.probe:
        caps = run_probe()
        print(json.dumps({m: {k: v for k, v in e.items() if k != "errors"}
                          for m, e in caps.items()}, indent=1))
        print(f"wrote {CAPS_PATH}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
