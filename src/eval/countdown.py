"""Exact, execution-free verifier for Countdown arithmetic answers."""

import ast
from collections import Counter
from fractions import Fraction
import json
import re


def verify_countdown(response: str, reference: str) -> bool:
    """Reference is JSON containing integer ``numbers`` and ``target``.

Each supplied number must appear exactly once. Only binary +, -, *, / and
parentheses are accepted. Python execution, exponentiation and new numbers
are never permitted. Fractions avoid floating-point acceptance tolerances.
"""
    task = json.loads(reference)
    numbers, target = task["numbers"], task["target"]
    if not isinstance(numbers, list) or not 1 <= len(numbers) <= 16 or not all(
        type(number) is int and 0 < number <= 10**9 for number in numbers
    ) or type(target) is not int:
        raise ValueError("invalid Countdown reference")
    match = re.search(r"<answer>\s*([^<>]+?)\s*</answer>\s*$", response, re.S)
    if not match or len(match.group(1)) > 256:
        return False
    used = []

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            used.append(node.value)
            return Fraction(node.value)
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            raise ValueError("unsupported arithmetic")
        left, right = evaluate(node.left), evaluate(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left / right

    try:
        tree = ast.parse(match.group(1).strip(), mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 100:
            return False
        value = evaluate(tree.body)
        return Counter(used) == Counter(numbers) and value == target
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError):
        return False
