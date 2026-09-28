"""Verification for the static_router baseline (build order step 4).

RouteLLM's own reported numbers (85% cost reduction, 95% GPT-4
performance) are on MT-Bench/MMLU/GSM8K with a GPT-4/Mixtral pair — not
comparable to a SWE-bench-based GCL run, so "matching reported numbers"
doesn't apply in the usual sense here. What IS checkable: RandomRouter's
output distribution matches RouteLLM's own definition (Uniform(0,1),
independent of the prompt) — verified statistically, not just by
reading the source.
"""

import random

import pytest

from src.models.cloud_llm import MockCloudLLM
from src.models.local_llm import MockLocalLLM
from src.models.router import RandomRouter, UnavailableRouter
from run import run_static_router


def test_random_router_win_rate_is_uniform_on_unit_interval():
    random.seed(0)
    router = RandomRouter()
    samples = [router.calculate_strong_win_rate("irrelevant") for _ in range(5000)]

    assert all(0.0 <= s <= 1.0 for s in samples)
    mean = sum(samples) / len(samples)
    # Uniform(0,1) has mean 0.5; with n=5000 the sample mean should be
    # tight around that — this is the statistical check standing in for
    # "reproduces the reported behavior" since no cross-benchmark number
    # comparison is meaningful here.
    assert 0.47 < mean < 0.53


def test_random_router_routes_approximately_1_minus_threshold_to_strong():
    random.seed(1)
    router = RandomRouter()
    threshold = 0.7
    n = 5000
    strong_count = sum(1 for _ in range(n) if router.route("x", threshold) == "strong")

    expected = (1 - threshold) * n
    # Loose tolerance — this is a Bernoulli count, not an exact match.
    assert abs(strong_count - expected) < 0.05 * n


def test_run_static_router_produces_valid_records():
    random.seed(2)
    records = run_static_router(
        cfg={},
        router=RandomRouter(),
        strong_model=MockCloudLLM(fixed_latency_ms=100.0),
        weak_model=MockLocalLLM(fixed_latency_ms=5.0),
        threshold=0.5,
        problem_statements=["a", "b", "c", "d"],
    )

    assert len(records) == 4
    assert all(r["answered_by"] in ("local", "cloud") for r in records)
    for r in records:
        if r["answered_by"] == "cloud":
            assert r["latency_ms"] == 100.0
        else:
            assert r["latency_ms"] == 5.0


def test_unavailable_router_fails_loudly_not_silently():
    router = UnavailableRouter(name="mf")
    with pytest.raises(RuntimeError, match="huggingface"):
        router.calculate_strong_win_rate("x")
