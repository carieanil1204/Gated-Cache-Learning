import pytest

from src.models.retrieval_store import RetrievalStore


def test_empty_store_is_maximally_novel():
    store = RetrievalStore()
    assert store.nearest("anything") is None
    assert store.novelty("anything") == 1.0


def test_exact_match_has_similarity_one():
    store = RetrievalStore()
    store.add("fix the null pointer exception in parser")
    result = store.nearest("fix the null pointer exception in parser")

    assert result is not None
    assert result.similarity == pytest.approx(1.0)
    assert store.novelty("fix the null pointer exception in parser") == pytest.approx(0.0)


def test_unrelated_query_has_no_similarity():
    store = RetrievalStore()
    store.add("fix null pointer exception parser")
    result = store.nearest("update changelog release notes")  # no shared tokens at all

    assert result is not None
    assert result.similarity == 0.0


def test_nearest_picks_most_similar_of_several():
    store = RetrievalStore()
    store.add("update the changelog for the release")
    store.add("fix the null pointer exception in the parser module")
    result = store.nearest("null pointer exception in parser")

    assert "null pointer" in result.text
