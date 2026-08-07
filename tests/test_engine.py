import asyncio
import unittest

from verifierfuzz.engine import (
    audit_cases,
    audit_consistency_async,
    audit_metamorphic_async,
)
from verifierfuzz.integrations import CallableVerifier
from verifierfuzz.mutators import TextMutationSuite
from verifierfuzz.protocol import ScorePolicy, VerifierCase


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.case = VerifierCase(
            case_id="math",
            completion=r"\boxed{42}",
            reference=r"\boxed{42}",
        )
        self.target = CallableVerifier(
            lambda case: "42" in case.completion,
            policy=ScorePolicy.zero_one(),
        )
        self.reference = CallableVerifier(
            lambda case: case.completion.strip() == case.reference,
            policy=ScorePolicy.zero_one(),
        )

    def test_differential_audit_records_mutation_lineage(self):
        findings = audit_cases(
            [self.case],
            self.target,
            self.reference,
            mutators=[TextMutationSuite()],
            seed=7,
        )

        self.assertTrue(findings)
        self.assertTrue(all(finding.kind == "false_positive" for finding in findings))
        self.assertTrue(any(finding.mutation is not None for finding in findings))
        self.assertTrue(
            all(
                finding.mutation.parent_case_id == "math"
                for finding in findings
                if finding.mutation
            )
        )

    def test_minimizer_preserves_disagreement(self):
        findings = audit_cases(
            [
                VerifierCase(
                    case_id="long",
                    completion="irrelevant words answer 42 trailing words",
                    reference="42",
                )
            ],
            self.target,
            self.reference,
            minimize=True,
        )

        self.assertEqual(len(findings), 1)
        minimized = findings[0].minimized_completion
        self.assertIn("42", minimized)
        self.assertLess(len(minimized), len(findings[0].case.completion))

    def test_metamorphic_audit_reports_decision_drift(self):
        exact_target = CallableVerifier(
            lambda case: case.completion == case.reference,
            policy=ScorePolicy.zero_one(),
        )

        findings = asyncio.run(
            audit_metamorphic_async(
                [self.case],
                exact_target,
                [TextMutationSuite()],
            )
        )

        self.assertTrue(findings)
        self.assertTrue(all(item.kind == "metamorphic_drift" for item in findings))

    def test_consistency_reports_stochastic_target(self):
        decisions = iter([True, False, True, False])
        target = CallableVerifier(
            lambda case: next(decisions),
            policy=ScorePolicy.zero_one(),
        )

        findings = asyncio.run(
            audit_consistency_async([self.case], target, repeats=4)
        )

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].kind, "inconsistent")
        self.assertEqual(findings[0].reference_outcome.trace["pass_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
