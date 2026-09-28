import pytest

from src.eval.metrics import compute_metrics


def test_cloud_only_all_records_are_cloud_hits_zero():
    records = [{"answered_by": "cloud", "latency_ms": 50.0, "success": None} for _ in range(5)]
    m = compute_metrics(records)

    assert m.hit_rate == 0.0
    assert m.cloud_calls == 5
    assert m.mean_latency_ms == 50.0
    assert m.task_success_rate is None  # nothing resolved
    assert m.n_resolved == 0


def test_mixed_local_and_cloud_with_resolved_outcomes():
    records = [
        {"answered_by": "local", "latency_ms": 10.0, "success": True},
        {"answered_by": "local", "latency_ms": 10.0, "success": False},
        {"answered_by": "cloud", "latency_ms": 100.0, "success": True},
        {"answered_by": "cloud", "latency_ms": 100.0, "success": None},
    ]
    m = compute_metrics(records)

    assert m.hit_rate == 0.5
    assert m.cloud_calls == 2
    assert m.mean_latency_ms == 55.0
    assert m.n_resolved == 3
    assert m.task_success_rate == pytest.approx(2 / 3)


def test_empty_records_raises():
    with pytest.raises(ValueError):
        compute_metrics([])
