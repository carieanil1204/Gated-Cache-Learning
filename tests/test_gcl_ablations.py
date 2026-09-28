"""Verifies the two GCL ablations (build order step 5) actually differ
the way PROPOSAL.md says they should: retrieval_no_train never adapts
online; local_no_gate absorbs every cloud answer unconditionally."""

from src.models.cloud_llm import MockCloudLLM
from src.models.retrieval_store import RetrievalStore
from run import run_retrieval_no_train, run_local_no_gate


SIMILAR_QUERIES = [
    "fix null pointer exception in parser module",
    "fix null pointer exception in parser code",
    "fix null pointer exception parser bug",
]


def test_retrieval_no_train_never_adapts_even_on_repeated_similar_queries():
    store = RetrievalStore()
    records = run_retrieval_no_train(
        cfg={},
        store=store,
        cloud_model=MockCloudLLM(),
        novelty_threshold=0.5,
        problem_statements=SIMILAR_QUERIES,
    )

    # Store never grows -> every query escalates, even near-duplicates.
    assert all(r["answered_by"] == "cloud" for r in records)
    assert len(store.entries) == 0


def test_local_no_gate_absorbs_cloud_answers_and_starts_hitting_locally():
    store = RetrievalStore()
    records = run_local_no_gate(
        cfg={},
        store=store,
        cloud_model=MockCloudLLM(),
        novelty_threshold=0.5,
        problem_statements=SIMILAR_QUERIES,
    )

    # First query is novel (empty store) -> escalates and gets absorbed.
    assert records[0]["answered_by"] == "cloud"
    # Subsequent near-duplicate queries should now hit locally.
    assert records[1]["answered_by"] == "local"
    assert records[2]["answered_by"] == "local"
    assert len(store.entries) == 1  # only the one cloud escalation was stored


def test_local_no_gate_absorbs_even_when_cloud_answer_would_be_wrong():
    """The whole point of this ablation: there is no quality check, so
    a bad cloud escalation gets memorized exactly like a good one —
    this is the mechanism, not a claim that the mock's answers are
    actually wrong (no oracle exists to judge that yet)."""
    store = RetrievalStore()
    run_local_no_gate(
        cfg={},
        store=store,
        cloud_model=MockCloudLLM(),
        novelty_threshold=0.5,
        problem_statements=["any query at all"],
    )
    assert len(store.entries) == 1  # admitted with zero filtering
