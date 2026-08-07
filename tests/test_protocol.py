import unittest

from verifierfuzz.protocol import (
    Decision,
    ScorePolicy,
    VerifierCase,
    outcome_from_raw,
)


class ProtocolTests(unittest.TestCase):
    def test_signed_policy_distinguishes_service_error(self):
        policy = ScorePolicy.signed()

        self.assertEqual(policy.classify(1), Decision.PASS)
        self.assertEqual(policy.classify(-1), Decision.FAIL)
        self.assertEqual(policy.classify(-2), Decision.ERROR)

    def test_mapping_response_preserves_message(self):
        outcome = outcome_from_raw(
            {"score": 1, "msg": "accepted"},
            ScorePolicy.zero_one(),
        )

        self.assertEqual(outcome.decision, Decision.PASS)
        self.assertEqual(outcome.trace["message"], "accepted")

    def test_unknown_mapping_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "score"):
            outcome_from_raw({"reward": 1}, ScorePolicy.zero_one())

    def test_tensor_like_scalar_does_not_require_framework_dependency(self):
        class Scalar:
            def item(self):
                return 1

        outcome = outcome_from_raw(Scalar(), ScorePolicy.zero_one())

        self.assertEqual(outcome.decision, Decision.PASS)
        self.assertEqual(outcome.score, 1.0)

    def test_with_completion_preserves_case_and_extends_metadata(self):
        case = VerifierCase(
            case_id="one",
            completion="before",
            metadata={"source": "seed"},
        )

        changed = case.with_completion("after", mutation="suffix")

        self.assertEqual(changed.case_id, "one")
        self.assertEqual(changed.completion, "after")
        self.assertEqual(changed.metadata["source"], "seed")
        self.assertEqual(changed.metadata["mutation"], "suffix")


if __name__ == "__main__":
    unittest.main()
