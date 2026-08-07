"""SARIF output for CI code-scanning artifact upload."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from verifierfuzz.engine import AuditFinding

from .serialize import finding_to_dict


def build_sarif(findings: Iterable[AuditFinding]) -> dict:
    finding_list = list(findings)
    kinds = sorted({finding.kind for finding in finding_list})
    rules = [
        {
            "id": kind,
            "name": kind.replace("_", " ").title(),
            "shortDescription": {
                "text": f"VerifierFuzz detected {kind.replace('_', ' ')}"
            },
        }
        for kind in kinds
    ]
    results = []
    for finding in finding_list:
        result = {
            "ruleId": finding.kind,
            "level": "error" if finding.kind == "false_positive" else "warning",
            "message": {
                "text": (
                    f"{finding.kind} for case {finding.case.case_id} "
                    f"under {finding.relation}"
                )
            },
            "properties": finding_to_dict(finding),
        }
        source = finding.case.metadata.get("source")
        if source:
            result["locations"] = [
                {
                    "physicalLocation": {
                        "artifactLocation": {"uri": str(source)},
                    }
                }
            ]
        results.append(result)
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "VerifierFuzz",
                        "informationUri": "https://github.com/XFWang522/VerifierFuzz",
                        "rules": rules,
                    }
                },
                "results": results,
            }
        ],
    }


def write_sarif(path: str, findings: Iterable[AuditFinding]) -> None:
    Path(path).write_text(
        json.dumps(build_sarif(findings), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
