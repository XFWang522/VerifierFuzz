"""Property-based testing for RL reward verifiers."""

from .core import Finding, audit, mutate_text
from .engine import (
    AuditFinding,
    audit_cases,
    audit_cases_async,
    audit_consistency_async,
    audit_metamorphic_async,
)
from .protocol import (
    AsyncVerifier,
    BatchVerifier,
    Decision,
    ScorePolicy,
    SyncVerifier,
    VerifierCase,
    VerifierOutcome,
)

__all__ = [
    "AsyncVerifier",
    "AuditFinding",
    "BatchVerifier",
    "Decision",
    "Finding",
    "ScorePolicy",
    "SyncVerifier",
    "VerifierCase",
    "VerifierOutcome",
    "audit",
    "audit_cases",
    "audit_cases_async",
    "audit_consistency_async",
    "audit_metamorphic_async",
    "mutate_text",
]
__version__ = "0.1.0"
