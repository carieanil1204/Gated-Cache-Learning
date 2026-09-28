"""Static router baseline — RouteLLM (Ong et al., arXiv:2406.18665).

Interface and RandomRouter are a faithful reproduction of RouteLLM's
own code (github.com/lm-sys/RouteLLM, routellm/routers/routers.py,
checked directly): `Router.calculate_strong_win_rate(prompt)` returns a
float in [0,1]; `route()` sends the query to the strong model iff that
win rate >= threshold. RandomRouter returns `random.uniform(0, 1)`,
ignoring the prompt entirely — verifiable by construction, not by
matching a benchmark number, since its output distribution is Uniform
independent of input.

RouteLLM's recommended router (`mf`, matrix factorization) and its
other trained routers (`bert`, `sw_ranking`, `causal_llm`) need
pretrained weights hosted on Hugging Face
(https://huggingface.co/routellm) — confirmed BLOCKED in this
environment (huggingface.co returns 403, policy denial, not a
transient failure — checked directly, see IMPLEMENTATION.md). Not
implemented here; NOT a shortcut, an environment limitation. RouteLLM's
own reported numbers (85% cost reduction / 95% GPT-4 performance) are
on MT-Bench/MMLU/GSM8K with a GPT-4/Mixtral pair — a different
benchmark entirely from SWE-bench, so those numbers were never going to
be directly comparable to a GCL run regardless of the HF block.
"""

import abc
import random
from dataclasses import dataclass


class Router(abc.ABC):
    @abc.abstractmethod
    def calculate_strong_win_rate(self, prompt: str) -> float: ...

    def route(self, prompt: str, threshold: float) -> str:
        """Returns "strong" (cloud) or "weak" (local)."""
        if self.calculate_strong_win_rate(prompt) >= threshold:
            return "strong"
        return "weak"


class RandomRouter(Router):
    """Exact reproduction of RouteLLM's random router."""

    def calculate_strong_win_rate(self, prompt: str) -> float:
        del prompt
        return random.uniform(0, 1)


@dataclass
class UnavailableRouter(Router):
    """Placeholder for mf/bert/sw_ranking/causal_llm — raises clearly
    rather than silently falling back to random if selected."""

    name: str

    def calculate_strong_win_rate(self, prompt: str) -> float:
        raise RuntimeError(
            f"RouteLLM's '{self.name}' router needs pretrained weights "
            "from huggingface.co/routellm, which is policy-blocked in "
            "this environment (confirmed via direct check, not assumed). "
            "Use RandomRouter here, or run this on an environment with "
            "HF Hub access."
        )
