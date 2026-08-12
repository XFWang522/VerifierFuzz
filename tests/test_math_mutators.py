import unittest
from random import Random

from verifierfuzz.mutators import MathMutationSuite
from verifierfuzz.protocol import VerifierCase


class MathMutationTests(unittest.TestCase):
    def test_math_suite_targets_answer_parser_boundaries(self):
        case = VerifierCase(
            case_id="math",
            completion=r"Reasoning.\n\boxed{42}",
            reference="42",
        )

        mutations = list(MathMutationSuite().mutate(case, Random(7)))
        by_name = {item.mutation.name: item for item in mutations}

        self.assertEqual(
            set(by_name),
            {
                "boxed-conflict-last",
                "boxed-conflict-first",
                "reference-leak",
                "trailing-reference",
                "multiple-final-markers",
                "malformed-box",
                "non-finite-prefix",
            },
        )
        self.assertTrue(
            by_name["boxed-conflict-last"].case.completion.endswith(
                r"\boxed{43}"
            )
        )
        self.assertTrue(
            by_name["trailing-reference"].case.completion.endswith("42")
        )

    def test_math_suite_records_deterministic_lineage(self):
        case = VerifierCase(
            case_id="math",
            completion="Final answer: 5-3=2",
            reference="2",
        )

        first = list(MathMutationSuite().mutate(case, Random(9)))
        second = list(MathMutationSuite().mutate(case, Random(9)))

        self.assertEqual(
            [item.mutation for item in first],
            [item.mutation for item in second],
        )
        self.assertTrue(
            all(
                item.case.metadata["mutation_suite"]
                == "math-reward-exploits"
                for item in first
            )
        )
        self.assertIn(
            "−",
            next(
                item.case.completion
                for item in first
                if item.mutation.name == "unicode-operators"
            ),
        )

    def test_non_numeric_reference_gets_distinct_wrong_answer(self):
        case = VerifierCase(
            case_id="symbolic",
            completion=r"\boxed{x}",
            reference="x",
        )

        mutations = list(MathMutationSuite().mutate(case, Random(0)))
        conflict = next(
            item
            for item in mutations
            if item.mutation.name == "boxed-conflict-last"
        )

        self.assertIn(r"\boxed{not-x}", conflict.case.completion)


if __name__ == "__main__":
    unittest.main()
