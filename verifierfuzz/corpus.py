"""Versioned JSONL case and regression corpora."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from verifierfuzz.engine import AuditFinding
from verifierfuzz.protocol import VerifierCase
from verifierfuzz.reporting.serialize import (
    case_from_dict,
    case_to_dict,
    finding_to_dict,
)


@dataclass(frozen=True)
class RegressionCase:
    case: VerifierCase
    expected_kind: Optional[str] = None


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error


def load_cases(path: str) -> List[VerifierCase]:
    cases = []
    for item in _read_jsonl(Path(path)):
        case_data = item.get("case", item)
        cases.append(case_from_dict(case_data))
    return cases


def load_regression_cases(path: str) -> List[RegressionCase]:
    entries = []
    for item in _read_jsonl(Path(path)):
        case_data = item.get("case", item)
        expected_kind = (
            item["expected_kind"]
            if "expected_kind" in item
            else item.get("kind")
        )
        entries.append(
            RegressionCase(
                case=case_from_dict(case_data),
                expected_kind=expected_kind,
            )
        )
    return entries


def write_cases(path: str, cases: Iterable[VerifierCase]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case_to_dict(case), ensure_ascii=False) + "\n")


def write_findings(path: str, findings: Iterable[AuditFinding]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for finding in findings:
            handle.write(
                json.dumps(finding_to_dict(finding), ensure_ascii=False) + "\n"
            )
