"""Pure aggregation over per-query outcome records — no model/data
dependencies, so these are testable in isolation. A record is a dict:
{"answered_by": "local" | "cloud", "latency_ms": float,
 "success": bool | None}  # None = unresolved (no oracle, no Judge J yet)
"""

from dataclasses import dataclass


@dataclass
class RunMetrics:
    hit_rate: float
    cloud_calls: int
    mean_latency_ms: float
    task_success_rate: float | None  # None if no record had a resolved outcome
    n_queries: int
    n_resolved: int


def compute_metrics(records: list[dict]) -> RunMetrics:
    if not records:
        raise ValueError("compute_metrics called with no records")

    n = len(records)
    hits = sum(1 for r in records if r["answered_by"] == "local")
    cloud_calls = sum(1 for r in records if r["answered_by"] == "cloud")
    mean_latency = sum(r["latency_ms"] for r in records) / n

    resolved = [r["success"] for r in records if r["success"] is not None]
    task_success_rate = (sum(resolved) / len(resolved)) if resolved else None

    return RunMetrics(
        hit_rate=hits / n,
        cloud_calls=cloud_calls,
        mean_latency_ms=mean_latency,
        task_success_rate=task_success_rate,
        n_queries=n,
        n_resolved=len(resolved),
    )
