"""M_C — the cloud LLM, per PROPOSAL.md's Components section.

CloudLLM is an interface, not a concrete API binding, so the cloud-only
baseline and later the gate's escalation path can be built and tested
without real API keys/network. AnthropicCloudLLM is the real backend,
wired to actual calls; MockCloudLLM is deterministic and used in tests
and local smoke runs.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CloudResponse:
    text: str
    latency_ms: float


class CloudLLM(ABC):
    @abstractmethod
    def answer(self, problem_statement: str) -> CloudResponse: ...


class MockCloudLLM(CloudLLM):
    """Deterministic stand-in for M_C — no network, no API key. Returns
    a fixed-format response so harness code (metrics, logging) can be
    exercised end-to-end before a real backend is wired in."""

    def __init__(self, fixed_latency_ms: float = 50.0):
        self.fixed_latency_ms = fixed_latency_ms
        self.call_count = 0

    def answer(self, problem_statement: str) -> CloudResponse:
        self.call_count += 1
        return CloudResponse(
            text=f"[mock patch for: {problem_statement[:40]}]",
            latency_ms=self.fixed_latency_ms,
        )


class AnthropicCloudLLM(CloudLLM):
    """Real backend. NOT YET USABLE in this environment — no API key
    configured here, no network call has been attempted or verified.
    Raises clearly on first use rather than silently no-op-ing."""

    def __init__(self, model: str, api_key_env: str = "ANTHROPIC_API_KEY"):
        import os

        self.model = model
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise RuntimeError(
                f"{api_key_env} not set — AnthropicCloudLLM cannot make "
                "real calls. Use MockCloudLLM for harness development, "
                "or set the key before running this baseline for real."
            )

    def answer(self, problem_statement: str) -> CloudResponse:
        raise NotImplementedError(
            "Real API call not yet implemented/verified in this "
            "environment. Wire up the actual client call here before "
            "using this class for a reported run."
        )
