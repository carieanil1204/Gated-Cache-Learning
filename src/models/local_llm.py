"""M_L — the local model, per PROPOSAL.md's Components section.
Same pattern as cloud_llm.py: an interface + a deterministic mock,
so harness/routing code is testable without a real small LM loaded."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LocalResponse:
    text: str
    latency_ms: float


class LocalLLM(ABC):
    @abstractmethod
    def answer(self, problem_statement: str) -> LocalResponse: ...


class MockLocalLLM(LocalLLM):
    """Deterministic stand-in for M_L. Lower fixed latency than
    MockCloudLLM by default, matching the local/cloud latency
    asymmetry the whole proposal is motivated by."""

    def __init__(self, fixed_latency_ms: float = 5.0):
        self.fixed_latency_ms = fixed_latency_ms
        self.call_count = 0

    def answer(self, problem_statement: str) -> LocalResponse:
        self.call_count += 1
        return LocalResponse(
            text=f"[mock local answer for: {problem_statement[:40]}]",
            latency_ms=self.fixed_latency_ms,
        )
