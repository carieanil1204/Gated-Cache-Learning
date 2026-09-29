"""Fang et al. GAPG baseline (arXiv:2509.24050, Algorithm 1).

This module keeps rollout evaluation and prompt filtering independent of a
particular language model. ``HuggingFaceFangPolicy`` implements the actual
sequence-level policy gradient in ``fang_hf.py``. A verifier is mandatory:
unverified model output must never receive a positive reward.
"""

from dataclasses import dataclass
import math
import random
import re
from typing import Callable, Protocol

from src.models.cloud_llm import CloudLLM


UNKNOWN_MARKER = "<unknown> I need external assistance </unknown>"
_HELP_AT_END = re.compile(r"<unknown>\s*I need external assistance\s*</unknown>\s*$", re.I)


def requests_help(response: str) -> bool:
    """Only a terminal help action invokes cloud; quoted markers do not."""
    return bool(_HELP_AT_END.search(response))


def verified_correct(verify: Callable[[str, str], bool], response: str, reference: str) -> bool:
    """An unavailable verifier is an error, not an incorrect training label."""
    result = verify(response, reference)
    if not isinstance(result, bool):
        raise TypeError("verifier must return bool; unresolved outcomes cannot train the policy")
    return result


@dataclass(frozen=True)
class FangExample:
    prompt: str
    reference_answer: str


@dataclass
class FangRollout:
    text: str
    # Preserve sampled tokens: decoding then tokenizing again changes likelihoods.
    prompt_ids: tuple[int, ...] = ()
    completion_ids: tuple[int, ...] = ()


@dataclass
class FangGroup:
    example: FangExample
    responses: list[FangRollout]
    rewards: list[float]
    local_correct: list[bool]
    cloud_correct: bool | None


class FangPolicy(Protocol):
    def sample(self, prompt: str, count: int) -> list[FangRollout]: ...
    def update(self, groups: list[FangGroup]) -> float: ...
    def answer(self, prompt: str, allow_help: bool = True) -> tuple[str, float]: ...


def group_coefficients(rewards: list[float]) -> list[float]:
    """Equation 2: (reward_i - group mean) / (G - 1), no std scaling."""
    if len(rewards) < 2:
        raise ValueError("GAPG requires group size >= 2")
    if not all(math.isfinite(reward) for reward in rewards):
        raise ValueError("rewards must be finite")
    mean = sum(rewards) / len(rewards)
    return [(reward - mean) / (len(rewards) - 1) for reward in rewards]


def score_group(
    example: FangExample,
    responses: list[FangRollout],
    cloud: CloudLLM,
    verify: Callable[[str, str], bool],
    accuracy_reward: float,
    coordination_reward: float,
) -> FangGroup:
    if not responses:
        raise ValueError("empty rollout group")
    if not math.isfinite(accuracy_reward) or not 0 <= coordination_reward < accuracy_reward:
        raise ValueError("require 0 <= coordination_reward < accuracy_reward")

    help_flags = [requests_help(response.text) for response in responses]
    local_correct = [
        False if help_requested else verified_correct(verify, response.text, example.reference_answer)
        for response, help_requested in zip(responses, help_flags)
    ]
    cloud_correct = None
    if any(help_flags):
        cloud_response = cloud.answer(example.prompt)  # at most once per prompt
        cloud_correct = verified_correct(verify, cloud_response.text, example.reference_answer)
    rewards = [
        coordination_reward if help_requested and cloud_correct else
        accuracy_reward if correct else 0.0
        for help_requested, correct in zip(help_flags, local_correct)
    ]
    return FangGroup(example, responses, rewards, local_correct, cloud_correct)


def select_groups(groups: list[FangGroup], rho: float, rng: random.Random) -> list[FangGroup]:
    """Section 3.3.2 D1/D2; retain constant groups in the batch denominator.

    Algorithm 1's short D1 description differs from Section 3.3.2. We follow
    the latter's explicit local-success definition; see docs/fang_baseline.md.
    """
    if not math.isfinite(rho) or rho < 0:
        raise ValueError("rho must be nonnegative")
    d1 = [
        group for group in groups
        if any(group.local_correct)
    ]
    d2 = [
        group for group in groups
        if not any(group.local_correct) and group.cloud_correct is True
    ]
    limit = min(len(d2), math.floor(rho * len(d1)))
    return d1 + rng.sample(d2, limit)


@dataclass(frozen=True)
class FangStep:
    sampled_prompts: int
    selected_local_prompts: int
    selected_cloud_prompts: int
    training_cloud_calls: int
    informative_prompts: int
    loss: float | None


def train_step(
    examples: list[FangExample],
    policy: FangPolicy,
    cloud: CloudLLM,
    verify: Callable[[str, str], bool],
    group_size: int,
    rho: float,
    accuracy_reward: float,
    coordination_reward: float,
    rng: random.Random,
) -> FangStep:
    if group_size < 2:
        raise ValueError("group_size must be >= 2")
    # Validate before sampling or incurring cloud calls.
    select_groups([], rho, rng)
    if not math.isfinite(accuracy_reward) or not 0 <= coordination_reward < accuracy_reward:
        raise ValueError("require 0 <= coordination_reward < accuracy_reward")
    groups = []
    for example in examples:
        responses = policy.sample(example.prompt, group_size)
        if len(responses) != group_size:
            raise ValueError("policy returned wrong group size")
        groups.append(score_group(
            example, responses, cloud, verify,
            accuracy_reward, coordination_reward,
        ))
    selected = select_groups(groups, rho, rng)
    loss = policy.update(selected) if selected else None
    local_count = sum(any(group.local_correct) for group in selected)
    return FangStep(
        sampled_prompts=len(groups),
        selected_local_prompts=local_count,
        selected_cloud_prompts=len(selected) - local_count,
        training_cloud_calls=sum(group.cloud_correct is not None for group in groups),
        informative_prompts=sum(min(group.rewards) < max(group.rewards) for group in selected),
        loss=loss,
    )


def exact_answer(response: str, reference: str) -> bool:
    """Simple verifier for answer datasets; never use for SWE-bench patches."""
    if requests_help(response):
        return False
    match = re.search(r"<answer>\s*(.*?)\s*</answer>\s*$", response, re.I | re.S)
    answer = match.group(1) if match else response
    return answer.strip() == reference.strip()


def run_fang(
    cfg: dict,
    policy: FangPolicy,
    cloud: CloudLLM,
    problem_statements: list[str],
) -> list[dict]:
    """Offline-trained policy, frozen during evaluation; fixed batch call cap.

    Excess help requests are regenerated locally with assistance disabled.
    Training calls are accounted for separately by ``FangStep``.
    """
    rho = cfg["fang"]["rho"]
    if not math.isfinite(rho) or rho < 0:
        raise ValueError("rho must be finite and nonnegative")
    budget = math.floor(len(problem_statements) * rho / (1 + rho) + 1e-12)
    cloud_calls = 0
    records = []
    for prompt in problem_statements:
        text, latency = policy.answer(prompt)
        help_requested = requests_help(text)
        denied = help_requested and cloud_calls >= budget
        answered_by = "local"
        if help_requested and not denied:
            response = cloud.answer(prompt)
            text = response.text
            latency += response.latency_ms
            cloud_calls += 1
            answered_by = "cloud"
        elif denied:
            text, retry_latency = policy.answer(prompt, allow_help=False)
            latency += retry_latency
        records.append({
            "answered_by": answered_by,
            "latency_ms": latency,
            "success": None,
            "response": text,
            "requested_help": help_requested,
            "budget_denied": denied,
        })
    return records


class MockFangPolicy:
    """Scripted routing smoke test; never presented as a trained policy."""

    def __init__(self):
        self.calls = 0

    def answer(self, prompt: str, allow_help: bool = True) -> tuple[str, float]:
        self.calls += 1
        if allow_help and self.calls % 2:
            return UNKNOWN_MARKER, 5.0
        return f"<answer>mock local answer: {prompt}</answer>", 5.0
