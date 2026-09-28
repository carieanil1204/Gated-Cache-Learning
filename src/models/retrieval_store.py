"""Retrieval store — part of M_L, per PROPOSAL.md's Components section.
Also the source of risk(x)'s "novelty" feature later (embedding
distance to nearest stored entry).

Embedding here is a plain bag-of-words vector + cosine similarity, not
a real sentence embedding model (that needs torch/transformers — more
dependency risk, and no GPU here to use one meaningfully anyway). This
is a deliberate placeholder for exercising retrieval/novelty *logic*
correctly (nearest-neighbor lookup, threshold behavior), not a claim
about real embedding quality — swap in a real embedding model before
trusting retrieval-quality numbers.
"""

import math
import re
from dataclasses import dataclass, field


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _bow_vector(tokens: list[str]) -> dict[str, int]:
    vec: dict[str, int] = {}
    for tok in tokens:
        vec[tok] = vec.get(tok, 0) + 1
    return vec


def _cosine_similarity(a: dict[str, int], b: dict[str, int]) -> float:
    shared = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in shared)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class RetrievedEntry:
    text: str
    similarity: float  # 1.0 = identical bag-of-words, 0.0 = nothing shared


@dataclass
class RetrievalStore:
    entries: list[str] = field(default_factory=list)
    _vectors: list[dict[str, int]] = field(default_factory=list)

    def add(self, text: str) -> None:
        self.entries.append(text)
        self._vectors.append(_bow_vector(_tokenize(text)))

    def nearest(self, query: str) -> RetrievedEntry | None:
        """Most similar stored entry, or None if the store is empty."""
        if not self.entries:
            return None
        query_vec = _bow_vector(_tokenize(query))
        best_idx, best_sim = 0, -1.0
        for i, vec in enumerate(self._vectors):
            sim = _cosine_similarity(query_vec, vec)
            if sim > best_sim:
                best_idx, best_sim = i, sim
        return RetrievedEntry(text=self.entries[best_idx], similarity=best_sim)

    def novelty(self, query: str) -> float:
        """1 - similarity to the nearest entry. 1.0 (maximally novel)
        for an empty store — see PROPOSAL.md's risk(x) classification:
        an empty store should never read as "low novelty"."""
        nearest = self.nearest(query)
        if nearest is None:
            return 1.0
        return 1.0 - nearest.similarity
