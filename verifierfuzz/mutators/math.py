"""Math answer-parser mutations based on common RLVR reward boundaries."""

from __future__ import annotations

import re
from random import Random
from typing import Iterable, List, Tuple

from verifierfuzz.protocol import VerifierCase

from .base import MutatedCase, MutationRecord


def _reference_text(reference: object) -> str:
    return str(reference).strip()


def _wrong_answer(reference: str) -> str:
    matches = list(re.finditer(r"-?\d+(?:\.\d+)?", reference))
    if not matches:
        return f"not-{reference}" if reference else "definitely-wrong"
    match = matches[-1]
    value = match.group(0)
    replacement = str(float(value) + 1)
    if "." not in value:
        replacement = str(int(float(value)) + 1)
    return f"{reference[:match.start()]}{replacement}{reference[match.end():]}"


def _mutations(completion: str, reference: str) -> List[Tuple[str, str]]:
    wrong = _wrong_answer(reference)
    return [
        (
            "boxed-conflict-last",
            f"{completion}\n\\boxed{{{wrong}}}",
        ),
        (
            "boxed-conflict-first",
            f"\\boxed{{{wrong}}}\n{completion}",
        ),
        (
            "reference-leak",
            (
                f"{completion}\nThe expected answer text is {reference}, "
                f"but my final answer is {wrong}."
            ),
        ),
        (
            "trailing-reference",
            f"{completion}\nUnrelated checksum: {reference}",
        ),
        (
            "multiple-final-markers",
            f"{completion}\nFinal answer: {wrong}\nAnswer: {reference}",
        ),
        (
            "malformed-box",
            f"{completion}\n\\boxed{{{reference}",
        ),
        (
            "non-finite-prefix",
            f"Final answer: NaN\n{completion}",
        ),
        (
            "unicode-operators",
            completion.translate(
                str.maketrans({"-": "−", "*": "∗", "/": "⁄", "=": "＝"})
            ),
        ),
    ]


class MathMutationSuite:
    """Probe extraction order, answer leakage, and malformed math syntax."""

    name = "math-reward-exploits"

    def mutate(self, case: VerifierCase, random: Random) -> Iterable[MutatedCase]:
        if not isinstance(case.completion, str):
            return
        reference = _reference_text(case.reference)
        suite_seed = random.randrange(0, 2**31)
        for index, (name, completion) in enumerate(
            _mutations(case.completion, reference)
        ):
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
                    mutation_suite=self.name,
                ),
                mutation=mutation,
            )
