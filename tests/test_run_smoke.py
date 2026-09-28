"""End-to-end smoke test for run.py's cloud_only path, per the
research-implementation skill's Step 2 (never launch a full run cold
— prove the harness runs and logs correctly on a small case first)."""

import csv

from src.eval.metrics import compute_metrics
from src.models.cloud_llm import MockCloudLLM
from run import run_cloud_only, append_run_log_row, get_git_commit


def test_run_cloud_only_produces_valid_records():
    model = MockCloudLLM(fixed_latency_ms=42.0)
    records = run_cloud_only(cfg={}, model=model, problem_statements=["a", "b", "c"])

    assert len(records) == 3
    assert all(r["answered_by"] == "cloud" for r in records)
    assert all(r["latency_ms"] == 42.0 for r in records)
    assert model.call_count == 3

    metrics = compute_metrics(records)
    assert metrics.hit_rate == 0.0
    assert metrics.cloud_calls == 3


def test_append_run_log_row_writes_header_once(tmp_path, monkeypatch):
    import run as run_module

    fake_log = tmp_path / "run_log.csv"
    monkeypatch.setattr(run_module, "RUN_LOG_PATH", fake_log)

    model = MockCloudLLM()
    records = run_cloud_only(cfg={}, model=model, problem_statements=["x"])
    metrics = compute_metrics(records)

    append_run_log_row("run1", {"condition": "cloud_only"}, 0, "completed", metrics, "abc123")
    append_run_log_row("run2", {"condition": "cloud_only"}, 1, "completed", metrics, "abc123")

    with open(fake_log) as f:
        rows = list(csv.reader(f))

    header_rows = [r for r in rows if r[0] == "run_id"]
    assert len(header_rows) == 1  # header written once, not per call
    assert len(rows) == 1 + 4 + 4  # header + 4 metrics per run x 2 runs


def test_get_git_commit_returns_a_real_hash():
    commit = get_git_commit()
    assert commit != "unknown"
    assert len(commit) == 40  # full SHA
