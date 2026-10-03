"""Stateful executable environment for V1 learning loop. Stdlib only.

Designed to mirror TRL's environment_factory contract:
  - reset(**kwargs) -> prompt str  (starts new episode)
  - public methods search_filing/open_section/get_table/submit_answer are tools
  - get_reward() -> float (environment-owned reward, used by GRPOTrainer)

Local harness drives it without any model: step() records calls, tracks
section cited, and scores on submit. TRL training script wraps this class.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from . import tools as T
from .reward import score_episode

DOCS = Path(__file__).resolve().parent.parent / "data" / "documents" / "filings.json"


class FilingEnv:
    """One task episode. Fresh instance per rollout (as TRL requires)."""

    def __init__(self, tasks: list[dict] | None = None, tasks_path: str | None = None):
        self._pool = tasks
        self.tasks_path = tasks_path
        self.task: dict | None = None
        self.tool_calls: list[dict] = []
        self.section_cited: str = "financial_statements"
        self._submission: dict | None = None

    def _sample(self) -> dict | None:
        if self._pool:
            return random.choice(self._pool)
        if self.tasks_path:
            lines = Path(self.tasks_path).read_text().splitlines()
            return json.loads(random.choice(lines))
        return None

    # -- TRL lifecycle --
    def reset(self, **kwargs) -> str | None:
        """Start episode. TRL calls reset() with no args -> self-sample.

        Local harness may pass task=... dict, or task_id + tasks_path lookup.
        """
        task = kwargs.get("task")
        if task is None and "task_id" in kwargs and self.tasks_path:
            for line in Path(self.tasks_path).read_text().splitlines():
                t = json.loads(line)
                if t["task_id"] == kwargs["task_id"]:
                    task = t
                    break
        if task is None:
            task = self._sample()
        if task is None:
            return None
        self.task = task
        self.tool_calls = []
        self.section_cited = "financial_statements"
        self._submission = None
        return (
            f"{task['prompt']}\n"
            "Use tools: search_filing(query), open_section(filing_id, section), "
            "get_table(filing_id, table_id). Then submit_answer(value, filing_id, evidence). "
            "Cite the exact evidence span."
        )

    def get_reward(self) -> float:
        """Environment-owned reward for TRL. Returns 0.0 if no submission."""
        if not self.task or not self._submission:
            return 0.0
        s = score_episode(self.task,
                          {"tool_calls": self.tool_calls, "section_cited": self.section_cited},
                          self._submission["value"], self._submission["filing_id"],
                          self._submission["evidence"])
        return s["reward"] / 10.0  # normalize to [0,1] for GRPO

    # -- tools (exposed to model) --
    def search_filing(self, query: str) -> str:
        """Search filings by company, ticker, or year.

        Args:
            query: Free-text query, e.g. "ACME 2024".

        Returns:
            Candidate filing_ids or NO_RESULTS.
        """
        r = T.search_filing(query)
        self.tool_calls.append({"tool": "search_filing", "result": r})
        return r

    def open_section(self, filing_id: str, section: str) -> str:
        """Open a section of a filing.

        Args:
            filing_id: Filing id, e.g. "ACME-10K-2024".
            section: One of financial_statements, mda, notes.

        Returns:
            Section text or ERROR.
        """
        r = T.open_section(filing_id, section)
        self.section_cited = section
        self.tool_calls.append({"tool": "open_section", "result": r})
        return r

    def get_table(self, filing_id: str, table_id: str) -> str:
        """Get a structured table from a filing.

        Args:
            filing_id: Filing id.
            table_id: One of income_statement, balance_sheet, cash_flow.

        Returns:
            JSON of metric values or ERROR.
        """
        r = T.get_table(filing_id, table_id)
        self.tool_calls.append({"tool": "get_table", "result": r})
        return r

    def submit_answer(self, value: str, filing_id: str, evidence: str) -> str:
        """Submit the final answer.

        Args:
            value: Extracted value, e.g. "$12,345 million".
            filing_id: Cited filing id.
            evidence: Exact evidence span from the section text.

        Returns:
            Confirmation string with score summary.
        """
        self._submission = {"value": value, "filing_id": filing_id, "evidence": evidence}
        s = score_episode(self.task,
                          {"tool_calls": self.tool_calls, "section_cited": self.section_cited},
                          value, filing_id, evidence)
        return f"SUBMITTED reward={s['reward']}/10 success={s['success']} breakdown={s['breakdown']}"
