"""Differential, metamorphic, and consistency auditing engines."""

from __future__ import annotations

import asyncio
import inspect
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, replace
from random import Random
from typing import Any, Awaitable, Callable, Iterable, List, Optional, Sequence

from verifierfuzz.mutators import MutationRecord, Mutator
from verifierfuzz.protocol import (
    Decision,
    VerifierCase,
    VerifierOutcome,
    error_outcome,
)


VerifierLike = Any


@dataclass(frozen=True)
class AuditFinding:
    case: VerifierCase
    target_outcome: VerifierOutcome
    reference_outcome: VerifierOutcome
    kind: str
    relation: str = "differential"
    mutation: Optional[MutationRecord] = None
    minimized_completion: Optional[str] = None


async def evaluate_verifier(
    verifier: VerifierLike, case: VerifierCase
) -> VerifierOutcome:
    """Evaluate sync or async verifier objects with exception isolation."""

    started = time.perf_counter()
    try:
        result = verifier.evaluate(case)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, VerifierOutcome):
            raise TypeError("verifier.evaluate() must return VerifierOutcome")
        if result.duration_ms == 0.0:
            result = replace(
                result,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        return result
    except Exception as error:
        return error_outcome(
            error,
            duration_ms=(time.perf_counter() - started) * 1000,
        )


def classify_disagreement(
    target: VerifierOutcome, reference: VerifierOutcome
) -> Optional[str]:
    if target.decision == Decision.ERROR:
        return "target_error"
    if reference.decision == Decision.ERROR:
        return "reference_error"
    if target.decision == reference.decision:
        return None
    if target.decision == Decision.PASS and reference.decision == Decision.FAIL:
        return "false_positive"
    if target.decision == Decision.FAIL and reference.decision == Decision.PASS:
        return "false_negative"
    return "decision_disagreement"


async def _minimize_text(
    case: VerifierCase,
    predicate: Callable[[VerifierCase], Awaitable[bool]],
) -> str:
    text = case.completion
    if not isinstance(text, str) or len(text) < 2:
        return text

    parts = re.findall(r"\S+\s*", text)
    if len(parts) < 2:
        parts = list(text)

    granularity = 2
    while len(parts) >= 2:
        chunk_size = int(math.ceil(len(parts) / granularity))
        reduced = False
        for start in range(0, len(parts), chunk_size):
            candidate_parts = parts[:start] + parts[start + chunk_size :]
            if not candidate_parts:
                continue
            candidate = "".join(candidate_parts)
            if await predicate(case.with_completion(candidate)):
                parts = candidate_parts
                granularity = max(2, granularity - 1)
                reduced = True
                break
        if reduced:
            continue
        if granularity >= len(parts):
            break
        granularity = min(len(parts), granularity * 2)
    return "".join(parts)


async def audit_cases_async(
    cases: Iterable[VerifierCase],
    target: VerifierLike,
    reference: VerifierLike,
    *,
    mutators: Sequence[Mutator] = (),
    seed: int = 0,
    include_seeds: bool = True,
    minimize: bool = False,
) -> List[AuditFinding]:
    """Run target/reference differential checks over seeds and mutations."""

    findings: List[AuditFinding] = []
    random = Random(seed)

    async def check(
        case: VerifierCase,
        mutation: Optional[MutationRecord],
    ) -> None:
        target_outcome, reference_outcome = await asyncio.gather(
            evaluate_verifier(target, case),
            evaluate_verifier(reference, case),
        )
        kind = classify_disagreement(target_outcome, reference_outcome)
        if kind is None:
            return

        minimized_completion = None
        if minimize and isinstance(case.completion, str):
            async def preserves_finding(candidate: VerifierCase) -> bool:
                candidate_target, candidate_reference = await asyncio.gather(
                    evaluate_verifier(target, candidate),
                    evaluate_verifier(reference, candidate),
                )
                return (
                    classify_disagreement(candidate_target, candidate_reference)
                    == kind
                )

            minimized_completion = await _minimize_text(case, preserves_finding)

        findings.append(
            AuditFinding(
                case=case,
                target_outcome=target_outcome,
                reference_outcome=reference_outcome,
                kind=kind,
                mutation=mutation,
                minimized_completion=minimized_completion,
            )
        )

    for case in cases:
        if include_seeds:
            await check(case, None)
        for mutator in mutators:
            generated = mutator.mutate(case, random)
            for mutated in generated:
                await check(mutated.case, mutated.mutation)
    return findings


def audit_cases(
    cases: Iterable[VerifierCase],
    target: VerifierLike,
    reference: VerifierLike,
    **kwargs: Any,
) -> List[AuditFinding]:
    return asyncio.run(audit_cases_async(cases, target, reference, **kwargs))


async def audit_metamorphic_async(
    cases: Iterable[VerifierCase],
    target: VerifierLike,
    mutators: Sequence[Mutator],
    *,
    seed: int = 0,
) -> List[AuditFinding]:
    """Report mutations that unexpectedly change the target decision."""

    findings: List[AuditFinding] = []
    random = Random(seed)
    for case in cases:
        seed_outcome = await evaluate_verifier(target, case)
        for mutator in mutators:
            for mutated in mutator.mutate(case, random):
                mutated_outcome = await evaluate_verifier(target, mutated.case)
                if (
                    Decision.ERROR not in (seed_outcome.decision, mutated_outcome.decision)
                    and seed_outcome.decision != mutated_outcome.decision
                ):
                    findings.append(
                        AuditFinding(
                            case=mutated.case,
                            target_outcome=mutated_outcome,
                            reference_outcome=seed_outcome,
                            kind="metamorphic_drift",
                            relation="metamorphic",
                            mutation=mutated.mutation,
                        )
                    )
    return findings


async def audit_consistency_async(
    cases: Iterable[VerifierCase],
    target: VerifierLike,
    *,
    repeats: int = 5,
) -> List[AuditFinding]:
    """Report stochastic verifier decisions with repeated-run statistics."""

    if repeats < 2:
        raise ValueError("repeats must be at least 2")
    findings: List[AuditFinding] = []
    for case in cases:
        outcomes = [await evaluate_verifier(target, case) for _ in range(repeats)]
        counts = Counter(outcome.decision.value for outcome in outcomes)
        non_error = [
            outcome for outcome in outcomes if outcome.decision != Decision.ERROR
        ]
        decisions = {outcome.decision for outcome in non_error}
        if len(decisions) <= 1:
            continue
        pass_count = sum(
            outcome.decision == Decision.PASS for outcome in non_error
        )
        pass_rate = pass_count / len(non_error)
        standard_error = math.sqrt(
            pass_rate * (1.0 - pass_rate) / len(non_error)
        )
        summary = VerifierOutcome(
            decision=Decision.ABSTAIN,
            trace={
                "counts": dict(counts),
                "pass_rate": pass_rate,
                "standard_error": standard_error,
                "repeats": repeats,
            },
        )
        findings.append(
            AuditFinding(
                case=case,
                target_outcome=outcomes[0],
                reference_outcome=summary,
                kind="inconsistent",
                relation="stochastic_consistency",
            )
        )
    return findings
