import unittest

from verifierfuzz.core import audit, mutate_text


class AuditTests(unittest.TestCase):
    def test_reports_false_positive(self):
        findings = audit(
            ["good", "good-but-wrong"],
            target=lambda value: value.startswith("good"),
            reference=lambda value: value == "good",
        )

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].candidate, "good-but-wrong")
        self.assertEqual(findings[0].kind, "false_positive")

    def test_reports_false_negative(self):
        findings = audit(
            ["valid"],
            target=lambda value: False,
            reference=lambda value: value == "valid",
        )

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].kind, "false_negative")

    def test_ignores_agreements(self):
        self.assertEqual(
            audit([1, 2], target=lambda value: True, reference=lambda value: True),
            [],
        )

    def test_text_mutations_are_deterministic(self):
        self.assertEqual(list(mutate_text("boxed")), list(mutate_text("boxed")))


if __name__ == "__main__":
    unittest.main()
