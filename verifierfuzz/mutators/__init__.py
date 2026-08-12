"""Built-in mutation suites."""

from .base import MutatedCase, MutationRecord, Mutator
from .math import MathMutationSuite
from .text import TextMutationSuite

__all__ = [
    "MathMutationSuite",
    "MutatedCase",
    "MutationRecord",
    "Mutator",
    "TextMutationSuite",
]
