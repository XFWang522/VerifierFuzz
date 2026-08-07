"""Framework-independent verifier contracts and score normalization."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple


class Decision(str, Enum):
    """Normalized verifier decision."""

    PASS = "pass"
    FAIL = "fail"
    ABSTAIN = "abstain"
    ERROR = "error"


@dataclass(frozen=True)
class VerifierCase:
    """One reward-verifier input independent of an RL framework."""

    case_id: str
    completion: Any
    prompt: Any = ""
    reference: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def with_completion(self, completion: Any, **metadata: Any) -> "VerifierCase":
        merged_metadata = dict(self.metadata)
        merged_metadata.update(metadata)
        return replace(self, completion=completion, metadata=merged_metadata)


@dataclass(frozen=True)
class VerifierOutcome:
    """Normalized result while preserving the verifier's raw response."""

    decision: Decision
    score: Optional[float] = None
    raw: Any = None
    trace: Mapping[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    error: Optional[str] = None


@dataclass(frozen=True)
class ScorePolicy:
    """Explicitly map framework-specific numeric rewards to decisions."""

    pass_threshold: float = 0.5
    pass_if_greater_equal: bool = True
    pass_values: Tuple[float, ...] = ()
    fail_values: Tuple[float, ...] = ()
    abstain_values: Tuple[float, ...] = ()
    error_values: Tuple[float, ...] = ()

    @classmethod
    def zero_one(cls) -> "ScorePolicy":
        return cls(pass_values=(1.0,), fail_values=(0.0,))

    @classmethod
    def signed(cls) -> "ScorePolicy":
        return cls(pass_values=(1.0,), fail_values=(-1.0,), error_values=(-2.0,))

    def classify(self, value: float) -> Decision:
        if value in self.error_values:
            return Decision.ERROR
        if value in self.abstain_values:
            return Decision.ABSTAIN
        if value in self.pass_values:
            return Decision.PASS
        if value in self.fail_values:
            return Decision.FAIL
        if self.pass_if_greater_equal:
            return Decision.PASS if value >= self.pass_threshold else Decision.FAIL
        return Decision.PASS if value > self.pass_threshold else Decision.FAIL


class SyncVerifier(Protocol):
    def evaluate(self, case: VerifierCase) -> VerifierOutcome:
        ...


class AsyncVerifier(Protocol):
    async def evaluate(self, case: VerifierCase) -> VerifierOutcome:
        ...


class BatchVerifier(Protocol):
    def evaluate_batch(
        self, cases: Sequence[VerifierCase]
    ) -> Sequence[VerifierOutcome]:
        ...


def outcome_from_raw(
    raw: Any,
    policy: ScorePolicy,
    *,
    duration_ms: float = 0.0,
    trace: Optional[Mapping[str, Any]] = None,
) -> VerifierOutcome:
    """Normalize bool, numeric, and common mapping reward responses."""

    if isinstance(raw, VerifierOutcome):
        return raw

    raw_trace: Dict[str, Any] = dict(trace or {})
    score_value = raw
    if isinstance(raw, Mapping):
        if "score" not in raw:
            raise ValueError("mapping verifier response must contain a 'score' field")
        score_value = raw["score"]
        if "msg" in raw:
            raw_trace.setdefault("message", raw["msg"])

    if not isinstance(score_value, (bool, int, float)) and hasattr(
        score_value, "item"
    ):
        score_value = score_value.item()

    if isinstance(score_value, bool):
        score = float(score_value)
        decision = Decision.PASS if score_value else Decision.FAIL
    elif isinstance(score_value, (int, float)):
        score = float(score_value)
        decision = policy.classify(score)
    else:
        raise TypeError(
            "verifier response must be bool, numeric, VerifierOutcome, "
            "or a mapping containing score"
        )

    return VerifierOutcome(
        decision=decision,
        score=score,
        raw=raw,
        trace=raw_trace,
        duration_ms=duration_ms,
    )


def error_outcome(error: BaseException, *, duration_ms: float = 0.0) -> VerifierOutcome:
    return VerifierOutcome(
        decision=Decision.ERROR,
        duration_ms=duration_ms,
        error=f"{type(error).__name__}: {error}",
    )
