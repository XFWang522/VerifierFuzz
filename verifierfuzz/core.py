"""Core differential auditing primitives."""

from dataclasses import dataclass
from typing import Callable, Generic, Iterable, List, TypeVar


Candidate = TypeVar("Candidate")
Verifier = Callable[[Candidate], bool]


@dataclass(frozen=True)
class Finding(Generic[Candidate]):
    """A decision disagreement between a target and reference verifier."""

    candidate: Candidate
    target_accepts: bool
    reference_accepts: bool

    @property
    def kind(self) -> str:
        if self.target_accepts:
            return "false_positive"
        return "false_negative"


def audit(
    candidates: Iterable[Candidate],
    target: Verifier[Candidate],
    reference: Verifier[Candidate],
) -> List[Finding[Candidate]]:
    """Return every candidate on which two verifier decisions disagree."""

    findings = []
    for candidate in candidates:
        target_accepts = target(candidate)
        reference_accepts = reference(candidate)
        if target_accepts != reference_accepts:
            findings.append(
                Finding(
                    candidate=candidate,
                    target_accepts=target_accepts,
                    reference_accepts=reference_accepts,
                )
            )
    return findings


def mutate_text(seed: str) -> Iterable[str]:
    """Generate deterministic baseline mutations for a textual completion."""

    yield seed
    yield f"prefix {seed}"
    yield f"{seed} suffix"
    yield f"{seed}0"
    yield seed.replace("boxed", "box")
    yield seed[: max(1, len(seed) // 2)]
