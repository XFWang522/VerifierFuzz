"""Adapter for ordinary Python reward callables."""

from __future__ import annotations

import inspect
import time
from typing import Any, Callable

from verifierfuzz.protocol import (
    ScorePolicy,
    VerifierCase,
    VerifierOutcome,
    outcome_from_raw,
)


class CallableVerifier:
    """Adapt ``fn(case)`` or ``fn(completion, reference)`` to the protocol."""

    def __init__(
        self,
        function: Callable[..., Any],
        *,
        policy: ScorePolicy = ScorePolicy(),
        pass_case: bool = True,
    ) -> None:
        self.function = function
        self.policy = policy
        self.pass_case = pass_case

    async def evaluate(self, case: VerifierCase) -> VerifierOutcome:
        started = time.perf_counter()
        if self.pass_case:
            raw = self.function(case)
        else:
            raw = self.function(case.completion, case.reference)
        if inspect.isawaitable(raw):
            raw = await raw
        return outcome_from_raw(
            raw,
            self.policy,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
