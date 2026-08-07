"""Mutation contracts with deterministic lineage."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any, Iterable, Mapping, Protocol

from verifierfuzz.protocol import VerifierCase


@dataclass(frozen=True)
class MutationRecord:
    name: str
    seed: int
    parent_case_id: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MutatedCase:
    case: VerifierCase
    mutation: MutationRecord


class Mutator(Protocol):
    name: str

    def mutate(self, case: VerifierCase, random: Random) -> Iterable[MutatedCase]:
        ...
