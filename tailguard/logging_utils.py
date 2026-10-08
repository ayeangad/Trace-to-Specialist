"""Append-only JSONL helpers + run-config sidecars. Stdlib only."""
from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def append_jsonl(path: str | Path, obj: dict) -> None:
    """Append one JSON object as a line; flush + fsync for resumability."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: str | Path) -> list[dict]:
    """Read all JSON lines; empty list if the file is missing."""
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def read_done_ids(path: str | Path, key: str = "task_id") -> set[str]:
    """Ids already present in a JSONL output (for restart skipping)."""
    return {str(r[key]) for r in read_jsonl(path) if key in r}


def write_config(out_path: str | Path, args_dict: dict) -> Path:
    """Write <out_path>.config.json with args, git hash, UTC time, Python."""
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_hash = "unknown"
    cfg = {
        "args": dict(args_dict),
        "git_hash": git_hash,
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
    }
    cfg_path = Path(str(out_path) + ".config.json")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=1) + "\n")
    return cfg_path
