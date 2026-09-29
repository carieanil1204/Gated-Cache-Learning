"""Method-level checks for Fang rewards, filtering and inference accounting."""

import random

import pytest

from src.models.cloud_llm import CloudLLM, CloudResponse
from src.models.fang import (
    FangExample, FangRollout, UNKNOWN_MARKER, exact_answer,
    group_coefficients, requests_help, run_fang, score_group, select_groups, train_step,
)


class Cloud(CloudLLM):
    def __init__(self, text="<answer>42</answer>"):
        self.text = text
        self.calls = 0

    def answer(self, prompt):
        self.calls += 1
        return CloudResponse(self.text, 20.0)


def group(responses, cloud=None):
    return score_group(
        FangExample("question", "42"), [FangRollout(text) for text in responses],
        cloud or Cloud(), exact_answer, 1.0, 0.25,
    )


def test_rewards_are_exclusive_and_teacher_is_called_once_per_group():
    cloud = Cloud()
    result = group(["<answer>42</answer>", "wrong", UNKNOWN_MARKER, UNKNOWN_MARKER], cloud)
    assert result.rewards == [1.0, 0.0, 0.25, 0.25]
    assert result.local_correct == [True, False, False, False]
    assert cloud.calls == 1


def test_no_reward_for_failed_cloud_and_no_call_without_help():
    cloud = Cloud("wrong")
    assert group([UNKNOWN_MARKER, "wrong"], cloud).rewards == [0.0, 0.0]
    group(["42", "wrong"], cloud)
    assert cloud.calls == 1


def test_unresolved_verifier_is_not_an_incorrect_training_label():
    with pytest.raises(TypeError, match="unresolved"):
        score_group(FangExample("q", "42"), [FangRollout("42")], Cloud(), lambda *_: None, 1.0, 0.25)


@pytest.mark.parametrize("text,expected", [
    (UNKNOWN_MARKER, True),
    ("<think>Uncertain</think>\n" + UNKNOWN_MARKER + "\n", True),
    ("Mention " + UNKNOWN_MARKER + " then give an answer", False),
    ("<unknown>maybe</unknown>", False),
    ("<unknown> I need external assistance", False),
])
def test_terminal_help_action(text, expected):
    assert requests_help(text) is expected


def test_coefficients_preserve_reward_scale_and_unbiased_factor():
    assert group_coefficients([1.0, 0.0]) == [0.5, -0.5]
    assert group_coefficients([0.25, 0.0]) == [0.125, -0.125]
    assert group_coefficients([1.0, 1.0]) == [0.0, 0.0]
    with pytest.raises(ValueError):
        group_coefficients([1.0])


def test_filter_ratio_and_disjoint_sets_with_constant_groups():
    local = [group(["42", "wrong"]) for _ in range(7)]
    assisted = [group([UNKNOWN_MARKER, "wrong"]) for _ in range(8)]
    failed = group(["wrong", "wrong"])
    selected = select_groups(local + assisted + [failed], 3 / 7, random.Random(7))
    assert len(selected) == 10
    assert sum(any(item.local_correct) for item in selected) == 7
    assert all(item is not failed for item in selected)
    # Section 3.3.2 includes all-correct groups in the D1 count.
    constant = group(["42", "42"])
    assert select_groups([constant, assisted[0]], 1.0, random.Random(0)) == [constant, assisted[0]]


def test_no_local_success_yields_no_update():
    class Policy:
        def sample(self, prompt, count):
            return [FangRollout(UNKNOWN_MARKER), FangRollout("wrong")]

        def update(self, groups):
            pytest.fail("empty selected batch must not update the model")

    cloud = Cloud()
    result = train_step([FangExample("q", "42")], Policy(), cloud, exact_answer, 2, 1.0, 1.0, 0.25, random.Random(0))
    assert result.loss is None
    assert result.training_cloud_calls == 1


def test_wrong_rollout_count_fails_before_cloud_call():
    class Policy:
        def sample(self, prompt, count):
            return [FangRollout(UNKNOWN_MARKER)]

    cloud = Cloud()
    with pytest.raises(ValueError, match="group size"):
        train_step([FangExample("q", "42")], Policy(), cloud, exact_answer, 2, 1.0, 1.0, 0.25, random.Random(0))
    assert cloud.calls == 0


def test_inference_caps_calls_and_includes_local_and_retry_latency():
    class Policy:
        def answer(self, prompt, allow_help=True):
            return (UNKNOWN_MARKER if allow_help else "42"), 5.0

    cloud = Cloud()
    records = run_fang({"fang": {"rho": 3 / 7}}, Policy(), cloud, ["q"] * 10)
    assert cloud.calls == 3
    assert [r["latency_ms"] for r in records] == [25.0] * 3 + [10.0] * 7
    assert sum(r["budget_denied"] for r in records) == 7
    assert all(r["response"] == "42" for r in records[3:])


def test_exact_verifier_does_not_grade_an_abstention_as_an_answer():
    assert exact_answer("<think>work</think><answer>42</answer>", "42")
    assert not exact_answer("<answer>42</answer>" + UNKNOWN_MARKER, "42")
    assert not exact_answer("<answer>42</answer> contradicting text", "42")
