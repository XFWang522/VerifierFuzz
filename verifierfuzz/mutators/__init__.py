"""Built-in mutation suites."""

from .base import MutatedCase, MutationRecord, Mutator
from .text import TextMutationSuite

__all__ = ["MutatedCase", "MutationRecord", "Mutator", "TextMutationSuite"]
