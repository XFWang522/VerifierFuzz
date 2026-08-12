"""Command-line entry point."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from .corpus import (
    load_cases,
    load_regression_cases,
    write_findings,
    write_regression_findings,
)
from .dataset import DatasetColumns, load_dataset_cases
from .engine import audit_cases
from .integrations import (
    CallableVerifier,
    SlimeBatchVerifier,
    SlimeVerifier,
    TrlVerifier,
    VerlVerifier,
)
from .mutators import MathMutationSuite, TextMutationSuite
from .protocol import ScorePolicy, VerifierCase
from .reporting import render_summary, write_sarif


def _weak_math_verifier(case: VerifierCase) -> bool:
    return "42" in case.completion


def _strict_math_reference(case: VerifierCase) -> bool:
    return case.completion.strip() == case.reference


def _load_symbol(spec: str) -> Any:
    module_spec, separator, symbol_name = spec.rpartition(":")
    if not separator:
        raise ValueError("import spec must use 'module:symbol' or '/path/file.py:symbol'")
    path = Path(module_spec)
    if path.suffix == ".py" or path.exists():
        module_name = f"verifierfuzz_user_{abs(hash(path.resolve()))}"
        import_spec = importlib.util.spec_from_file_location(module_name, path)
        if import_spec is None or import_spec.loader is None:
            raise ImportError(f"cannot import module from {path}")
        module = importlib.util.module_from_spec(import_spec)
        sys.modules[module_name] = module
        import_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_spec)
    return getattr(module, symbol_name)


def _score_policy(name: str, threshold: float) -> ScorePolicy:
    if name == "signed":
        return ScorePolicy.signed()
    if name == "zero-one":
        return ScorePolicy.zero_one()
    return ScorePolicy(pass_threshold=threshold)


def _load_verifier(spec: str, adapter: str, policy: ScorePolicy) -> Any:
    symbol = _load_symbol(spec)
    if hasattr(symbol, "evaluate"):
        return symbol
    if adapter == "verl":
        return VerlVerifier(symbol, policy=policy)
    if adapter == "trl":
        return TrlVerifier(symbol, policy=policy)
    if adapter == "slime":
        return SlimeVerifier(symbol, policy=policy)
    if adapter == "slime-group":
        return SlimeBatchVerifier(symbol, policy=policy)
    if adapter == "roll":
        raise ValueError(
            "ROLL CLI symbols must be configured RollVerifier instances"
        )
    return CallableVerifier(
        symbol,
        policy=policy,
        pass_case=adapter == "callable",
    )


def _write_outputs(args: argparse.Namespace, findings: Sequence[Any]) -> None:
    print(render_summary(findings))
    if args.output:
        write_findings(args.output, findings)
        print(f"JSONL: {args.output}")
    if args.sarif:
        write_sarif(args.sarif, findings)
        print(f"SARIF: {args.sarif}")


def run_demo() -> int:
    target = CallableVerifier(_weak_math_verifier, policy=ScorePolicy.zero_one())
    reference = CallableVerifier(
        _strict_math_reference,
        policy=ScorePolicy.zero_one(),
    )
    findings = audit_cases(
        [
            VerifierCase(
                case_id="demo-math",
                completion=r"\boxed{42}",
                reference=r"\boxed{42}",
            )
        ],
        target,
        reference,
        mutators=[TextMutationSuite()],
        include_seeds=True,
        minimize=True,
    )
    print(render_summary(findings))
    for finding in findings:
        completion = finding.minimized_completion or finding.case.completion
        print(f"[{finding.kind}] {completion!r}")
    return 0


def _audit_command(args: argparse.Namespace, *, mutations: bool) -> int:
    target = _load_verifier(
        args.target,
        args.adapter,
        _score_policy(args.target_policy, args.target_threshold),
    )
    reference = _load_verifier(
        args.oracle,
        args.oracle_adapter,
        _score_policy(args.oracle_policy, args.oracle_threshold),
    )
    findings = audit_cases(
        load_cases(args.corpus),
        target,
        reference,
        mutators=[TextMutationSuite()] if mutations else [],
        seed=args.seed,
        include_seeds=True,
        minimize=args.minimize,
    )
    _write_outputs(args, findings)
    return 1 if len(findings) > args.max_findings else 0


def _regression_command(args: argparse.Namespace) -> int:
    target = _load_verifier(
        args.target,
        args.adapter,
        _score_policy(args.target_policy, args.target_threshold),
    )
    reference = _load_verifier(
        args.oracle,
        args.oracle_adapter,
        _score_policy(args.oracle_policy, args.oracle_threshold),
    )
    mismatches = []
    all_findings = []
    for entry in load_regression_cases(args.corpus):
        findings = audit_cases(
            [entry.case],
            target,
            reference,
            include_seeds=True,
        )
        all_findings.extend(findings)
        actual_kind = findings[0].kind if findings else None
        if actual_kind != entry.expected_kind:
            mismatches.append(
                f"{entry.case.case_id}: expected={entry.expected_kind!r}, "
                f"actual={actual_kind!r}"
            )
    _write_outputs(args, all_findings)
    for mismatch in mismatches:
        print(f"REGRESSION {mismatch}")
    return 3 if mismatches else 0


def _scan_command(args: argparse.Namespace) -> int:
    target = _load_verifier(
        args.target,
        args.adapter,
        _score_policy(args.target_policy, args.target_threshold),
    )
    reference = _load_verifier(
        args.oracle,
        args.oracle_adapter,
        _score_policy(args.oracle_policy, args.oracle_threshold),
    )
    cases = load_dataset_cases(
        args.dataset,
        columns=DatasetColumns(
            prompt=args.prompt_column,
            completion=args.completion_column,
            reference=args.reference_column,
            case_id=args.id_column,
            metadata=args.metadata_column,
        ),
        format=args.format,
        framework=args.framework,
        offset=args.offset,
        limit=args.limit,
    )
    mutators = []
    if args.mutation_profile in ("text", "all"):
        mutators.append(TextMutationSuite())
    if args.mutation_profile in ("math", "all"):
        mutators.append(MathMutationSuite())
    findings = audit_cases(
        cases,
        target,
        reference,
        mutators=mutators,
        seed=args.seed,
        include_seeds=True,
        minimize=args.minimize,
    )
    _write_outputs(args, findings)
    if args.regression_output:
        write_regression_findings(args.regression_output, findings)
        print(f"Regression corpus: {args.regression_output}")
    return 1 if len(findings) > args.max_findings else 0


def _add_verifier_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target", required=True, help="Target import spec")
    parser.add_argument("--oracle", required=True, help="Reference import spec")
    parser.add_argument(
        "--adapter",
        choices=["callable", "pair", "verl", "trl", "slime", "slime-group", "roll"],
        default="callable",
    )
    parser.add_argument(
        "--oracle-adapter",
        choices=["callable", "pair", "verl", "trl", "slime", "slime-group", "roll"],
        default="callable",
    )
    parser.add_argument(
        "--target-policy",
        choices=["threshold", "signed", "zero-one"],
        default="threshold",
    )
    parser.add_argument(
        "--oracle-policy",
        choices=["threshold", "signed", "zero-one"],
        default="threshold",
    )
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--oracle-threshold", type=float, default=0.5)


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", help="Write findings as JSONL")
    parser.add_argument("--sarif", help="Write findings as SARIF 2.1.0")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verifierfuzz",
        description="Find reward exploits before your model learns them.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("demo", help="Run the executable proof of concept")

    scan_parser = subparsers.add_parser(
        "scan",
        help="Audit reward functions against a JSONL, JSON, or Parquet dataset",
    )
    _add_verifier_arguments(scan_parser)
    _add_output_arguments(scan_parser)
    scan_parser.add_argument("--dataset", required=True)
    scan_parser.add_argument(
        "--format",
        choices=["auto", "jsonl", "json", "parquet"],
        default="auto",
    )
    scan_parser.add_argument(
        "--framework",
        choices=["generic", "verl", "slime", "roll", "trl"],
        default="generic",
    )
    scan_parser.add_argument("--prompt-column", default="prompt")
    scan_parser.add_argument("--completion-column", default="response")
    scan_parser.add_argument("--reference-column", default="ground_truth")
    scan_parser.add_argument("--id-column", default="id")
    scan_parser.add_argument(
        "--metadata-column",
        action="append",
        default=[],
        help="Copy a dotted dataset field into case metadata; repeatable",
    )
    scan_parser.add_argument("--offset", type=int, default=0)
    scan_parser.add_argument("--limit", type=int)
    scan_parser.add_argument("--seed", type=int, default=0)
    scan_parser.add_argument(
        "--mutation-profile",
        choices=["none", "text", "math", "all"],
        default="all",
    )
    scan_parser.add_argument("--minimize", action="store_true")
    scan_parser.add_argument("--max-findings", type=int, default=0)
    scan_parser.add_argument(
        "--regression-output",
        help="Freeze current findings into a regression JSONL corpus",
    )

    audit_parser = subparsers.add_parser(
        "audit",
        help="Mutate a corpus and compare target with reference",
    )
    _add_verifier_arguments(audit_parser)
    _add_output_arguments(audit_parser)
    audit_parser.add_argument("--corpus", required=True)
    audit_parser.add_argument("--seed", type=int, default=0)
    audit_parser.add_argument("--minimize", action="store_true")
    audit_parser.add_argument("--max-findings", type=int, default=0)

    replay_parser = subparsers.add_parser(
        "replay",
        help="Replay a corpus without generating mutations",
    )
    _add_verifier_arguments(replay_parser)
    _add_output_arguments(replay_parser)
    replay_parser.add_argument("--corpus", required=True)
    replay_parser.add_argument("--seed", type=int, default=0)
    replay_parser.add_argument("--minimize", action="store_true")
    replay_parser.add_argument("--max-findings", type=int, default=0)

    regression_parser = subparsers.add_parser(
        "regression",
        help="Assert expected disagreement kinds in a corpus",
    )
    _add_verifier_arguments(regression_parser)
    _add_output_arguments(regression_parser)
    regression_parser.add_argument("--corpus", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        return run_demo()
    if args.command == "scan":
        return _scan_command(args)
    if args.command == "audit":
        return _audit_command(args, mutations=True)
    if args.command == "replay":
        return _audit_command(args, mutations=False)
    if args.command == "regression":
        return _regression_command(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
