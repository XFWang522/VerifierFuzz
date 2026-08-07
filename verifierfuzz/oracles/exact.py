"""Small deterministic reference oracles."""

from __future__ import annotations

from typing import Callable

from verifierfuzz.protocol import (
    Decision,
    VerifierCase,
    VerifierOutcome,
)


class ExactMatchOracle:
    def __init__(
        self,
        *,
        normalize: Callable[[str], str] = lambda value: value.strip(),
    ) -> None:
        self._normalize = normalize

    def evaluate(self, case: VerifierCase) -> VerifierOutcome:
        if not isinstance(case.completion, str) or not isinstance(case.reference, str):
            raise TypeError("ExactMatchOracle requires string completion and reference")
        accepted = self._normalize(case.completion) == self._normalize(case.reference)
        return VerifierOutcome(
            decision=Decision.PASS if accepted else Decision.FAIL,
            score=float(accepted),
            raw=accepted,
        )
