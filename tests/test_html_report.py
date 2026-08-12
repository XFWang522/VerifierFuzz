import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from verifierfuzz.engine import AuditFinding
from verifierfuzz.protocol import Decision, VerifierCase, VerifierOutcome
from verifierfuzz.reporting import build_html, write_html


class HtmlReportTests(unittest.TestCase):
    def test_report_is_self_contained_filterable_and_escaped(self):
        finding = AuditFinding(
            case=VerifierCase(
                case_id="xss",
                prompt="<script>alert('prompt')</script>",
                completion="<img src=x onerror=alert(1)>",
                reference="safe",
                metadata={"source": "<unsafe>"},
            ),
            target_outcome=VerifierOutcome(
                decision=Decision.PASS,
                score=1.0,
            ),
            reference_outcome=VerifierOutcome(
                decision=Decision.FAIL,
                score=0.0,
            ),
            kind="false_positive",
            minimized_completion="<b>minimal</b>",
        )

        report = build_html(
            [finding],
            generated_at=datetime(2026, 8, 12, tzinfo=timezone.utc),
        )

        self.assertIn("<!doctype html>", report)
        self.assertIn('data-kind="false_positive"', report)
        self.assertIn("filterFindings", report)
        self.assertIn("&lt;script&gt;", report)
        self.assertIn("&lt;b&gt;minimal&lt;/b&gt;", report)
        self.assertNotIn("<script>alert('prompt')</script>", report)
        self.assertIn("2026-08-12T00:00:00+00:00", report)

    def test_empty_report_has_explicit_success_state(self):
        report = build_html([])

        self.assertIn("No disagreements found.", report)
        self.assertIn("0 findings", report)

    def test_write_html_creates_standalone_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "report.html"

            write_html(str(path), [])

            self.assertTrue(path.read_text(encoding="utf-8").endswith("</html>\n"))


if __name__ == "__main__":
    unittest.main()
