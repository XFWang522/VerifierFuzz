"""Thread-safe reporting sinks and human-readable summaries."""

from __future__ import annotations

import json
import threading
from collections import Counter
from pathlib import Path
from typing import Iterable

from verifierfuzz.engine import AuditFinding

from .serialize import finding_to_dict


class JsonlSink:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def __call__(self, finding: AuditFinding) -> None:
        line = json.dumps(finding_to_dict(finding), ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def render_summary(findings: Iterable[AuditFinding]) -> str:
    finding_list = list(findings)
    by_kind = Counter(finding.kind for finding in finding_list)
    by_relation = Counter(finding.relation for finding in finding_list)
    lines = [f"findings: {len(finding_list)}"]
    if by_kind:
        lines.append(
            "by kind: "
            + ", ".join(f"{key}={value}" for key, value in sorted(by_kind.items()))
        )
    if by_relation:
        lines.append(
            "by relation: "
            + ", ".join(
                f"{key}={value}" for key, value in sorted(by_relation.items())
            )
        )
    return "\n".join(lines)
