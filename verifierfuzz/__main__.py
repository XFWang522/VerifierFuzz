"""Command-line entry point."""

import argparse
from typing import Sequence

from .core import audit, mutate_text


def weak_math_verifier(candidate: str) -> bool:
    """Intentionally vulnerable verifier used by the local demo."""

    return "42" in candidate


def strict_math_reference(candidate: str) -> bool:
    """Reference contract requiring one exact final answer."""

    return candidate.strip() == r"\boxed{42}"


def run_demo() -> int:
    candidates = list(mutate_text(r"\boxed{42}"))
    findings = audit(candidates, weak_math_verifier, strict_math_reference)

    print(f"Audited {len(candidates)} candidates; found {len(findings)} disagreements.")
    for finding in findings:
        print(f"[{finding.kind}] {finding.candidate!r}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verifierfuzz",
        description="Find reward exploits before your model learns them.",
    )
    parser.add_argument(
        "command",
        choices=["demo"],
        help="Run the executable proof of concept.",
    )
    return parser


def main(argv: Sequence[str] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        return run_demo()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
