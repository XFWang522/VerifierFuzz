"""Stable JSON serialization for audit artifacts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from verifierfuzz.engine import AuditFinding
from verifierfuzz.protocol import Decision, VerifierCase, VerifierOutcome


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return repr(value)


def case_to_dict(case: VerifierCase) -> dict:
    return {
        "case_id": case.case_id,
        "prompt": json_safe(case.prompt),
        "completion": json_safe(case.completion),
        "reference": json_safe(case.reference),
        "metadata": json_safe(case.metadata),
    }


def case_from_dict(data: Mapping[str, Any]) -> VerifierCase:
    return VerifierCase(
        case_id=str(data["case_id"]),
        prompt=data.get("prompt", ""),
        completion=data.get("completion", ""),
        reference=data.get("reference"),
        metadata=data.get("metadata", {}),
    )


def outcome_to_dict(outcome: VerifierOutcome) -> dict:
    return {
        "decision": outcome.decision.value,
        "score": outcome.score,
        "raw": json_safe(outcome.raw),
        "trace": json_safe(outcome.trace),
        "duration_ms": outcome.duration_ms,
        "error": outcome.error,
    }


def outcome_from_dict(data: Mapping[str, Any]) -> VerifierOutcome:
    return VerifierOutcome(
        decision=Decision(str(data["decision"])),
        score=data.get("score"),
        raw=data.get("raw"),
        trace=data.get("trace", {}),
        duration_ms=float(data.get("duration_ms", 0.0)),
        error=data.get("error"),
    )


def finding_to_dict(finding: AuditFinding) -> dict:
    return {
        "schema_version": 1,
        "kind": finding.kind,
        "relation": finding.relation,
        "case": case_to_dict(finding.case),
        "target": outcome_to_dict(finding.target_outcome),
        "reference": outcome_to_dict(finding.reference_outcome),
        "mutation": json_safe(finding.mutation),
        "minimized_completion": finding.minimized_completion,
    }
