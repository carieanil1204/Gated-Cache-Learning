import json

import pytest

from src.eval.countdown import verify_countdown


def reference(numbers, target):
    return json.dumps({"numbers": numbers, "target": target})


def test_accepts_equivalent_expressions_and_fractional_intermediates():
    task = reference([3, 3, 8, 8], 24)
    assert verify_countdown("<answer>8 / (3 - 8 / 3)</answer>", task)
    assert verify_countdown("<answer>(8)/(3-(8/3))</answer>", task)


@pytest.mark.parametrize("expression", [
    "24", "8 + 8 + 8", "3 ** 3 - 3", "__import__('os').system('true')",
    "8 / (3 - 3) + 8", "8 + 8 + True + 7", "24.0",
])
def test_rejects_invalid_number_use_and_non_arithmetic(expression):
    assert not verify_countdown(f"<answer>{expression}</answer>", reference([3, 3, 8, 8], 24))


def test_requires_final_answer_and_valid_reference():
    assert not verify_countdown("<answer>1+2</answer><unknown>help</unknown>", reference([1, 2], 3))
    with pytest.raises(ValueError):
        verify_countdown("<answer>1+2</answer>", reference([1, True], 3))
