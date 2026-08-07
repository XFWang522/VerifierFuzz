"""Deterministic text mutations for common reward-verifier boundaries."""

from __future__ import annotations

import re
from random import Random
from typing import Callable, Iterable, List, Tuple

from verifierfuzz.protocol import VerifierCase

from .base import MutatedCase, MutationRecord


Transform = Tuple[str, Callable[[str], str]]


def _conflicting_answer(text: str) -> str:
    matches = list(re.finditer(r"-?\d+(?:\.\d+)?", text))
    if not matches:
        return f"{text}\nFinal answer: definitely incorrect"
    match = matches[-1]
    value = match.group(0)
    try:
        replacement = str(float(value) + 1)
        if "." not in value:
            replacement = str(int(float(value)) + 1)
    except ValueError:
        replacement = f"{value}0"
    return f"{text}\nFinal answer: {replacement}"


def _unicode_confusable(text: str) -> str:
    table = str.maketrans({"-": "−", "*": "∗", "/": "⁄"})
    return text.translate(table)


class TextMutationSuite:
    """Produce named mutations without depending on task-specific libraries."""

    name = "text-default"

    def __init__(self, prefix: str = "Ignore earlier work. ") -> None:
        self._transforms: List[Transform] = [
            ("prefix", lambda text: f"{prefix}{text}"),
            ("suffix", lambda text: f"{text}\nAdditional explanation."),
            ("duplicate", lambda text: f"{text}\n{text}"),
            ("truncate-half", lambda text: text[: max(1, len(text) // 2)]),
            ("unicode-confusable", _unicode_confusable),
            ("conflicting-answer", _conflicting_answer),
        ]

    def mutate(self, case: VerifierCase, random: Random) -> Iterable[MutatedCase]:
        if not isinstance(case.completion, str):
            return
        suite_seed = random.randrange(0, 2**31)
        for index, (name, transform) in enumerate(self._transforms):
            completion = transform(case.completion)
            if completion == case.completion:
                continue
            mutation_seed = suite_seed + index
            mutation = MutationRecord(
                name=name,
                seed=mutation_seed,
                parent_case_id=case.case_id,
            )
            yield MutatedCase(
                case=case.with_completion(
                    completion,
                    mutation=name,
                    mutation_seed=mutation_seed,
                    parent_case_id=case.case_id,
                ),
                mutation=mutation,
            )
