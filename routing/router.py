"""Task -> model router stub (V1). Full optimizer lands in V2.

V1 contract: cheapest model clearing the quality threshold.
Threshold default 90% task success. Model matrix is filled by experiments/.
"""
from __future__ import annotations


def route(task_family: str, model_matrix: dict[str, float],
          costs: dict[str, float], threshold: float = 0.90) -> str | None:
    """Return cheapest model with success >= threshold, else None (keep frontier)."""
    passing = {m: costs.get(m, float("inf")) for m, s in model_matrix.items() if s >= threshold}
    if not passing:
        return None
    return min(passing, key=lambda m: passing[m])


if __name__ == "__main__":
    # demo: 0.8B fails threshold, 2B passes cheaper than frontier
    m = {"qwen3.5-0.8b": 0.84, "qwen-2b": 0.91, "frontier": 0.95}
    c = {"qwen3.5-0.8b": 0.003, "qwen-2b": 0.008, "frontier": 0.040}
    print(route("financial_metric_extraction", m, c))
